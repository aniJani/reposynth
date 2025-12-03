use serde::{Deserialize, Serialize};
// use std::collections::HashMap;
// use std::io::{self, BufRead, Write};
// use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tree_sitter::{Node, Parser};

// Define the JSONL protocol for communication
#[derive(Serialize, Deserialize, Debug)]
struct Request {
    id: String,
    path: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Response {
    id: String,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ast: Option<Vec<AstNode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct AstNode {
    id: usize,
    kind: String,
    start_byte: usize,
    end_byte: usize,
    start_point: (usize, usize),
    end_point: (usize, usize),
    children: Vec<usize>,
}

// External language definitions from our build script
extern "C" {
    fn tree_sitter_python() -> tree_sitter::Language;
    fn tree_sitter_typescript() -> tree_sitter::Language;
    fn tree_sitter_css() -> tree_sitter::Language;
    fn tree_sitter_scss() -> tree_sitter::Language;
    fn tree_sitter_html() -> tree_sitter::Language;
}

// The main async function to run our daemon
#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let request: Request = match serde_json::from_str(&line) {
                    Ok(req) => req,
                    Err(e) => {
                        let response = Response {
                            id: "unknown".to_string(),
                            path: "unknown".to_string(),
                            ast: None,
                            error: Some(format!("Failed to parse request: {}", e)),
                        };
                        let response_json = serde_json::to_string(&response).unwrap();
                        stdout.write_all(format!("{}\n", response_json).as_bytes()).await.unwrap();
                        stdout.flush().await.unwrap();
                        line.clear();
                        continue;
                    }
                };

                // Use spawn_blocking to move CPU-intensive parsing off the Tokio runtime
                let response = tokio::task::spawn_blocking(move || {
                    process_file(request)
                }).await.unwrap();
                
                let response_json = serde_json::to_string(&response).unwrap();
                stdout.write_all(format!("{}\n", response_json).as_bytes()).await.unwrap();
                stdout.flush().await.unwrap();
                line.clear();
            }
            Err(e) => {
                // Handle read error, maybe log to stderr and exit
                eprintln!("Error reading from stdin: {}", e);
                break;
            }
        }
    }
}

// This function runs in a separate thread pool managed by spawn_blocking
fn process_file(request: Request) -> Response {
    let path = &request.path;
    let language = match path.split('.').last() {
        Some("py") => unsafe { tree_sitter_python() },
        Some("ts") | Some("tsx") | Some("js") | Some("jsx") => unsafe { tree_sitter_typescript() },
        Some("css") => unsafe { tree_sitter_css() },
        Some("scss") => unsafe { tree_sitter_scss() },
        Some("html") | Some("htm") => unsafe { tree_sitter_html() },
        _ => {
            return Response {
                id: request.id,
                path: request.path,
                ast: None,
                error: Some("Unsupported file type".to_string()),
            };
        }
    };

    let mut parser = Parser::new();
    parser.set_language(&language).unwrap();

    let source_code = match std::fs::read_to_string(path) {
        Ok(code) => code,
        Err(e) => {
            return Response {
                id: request.id,
                path: request.path,
                ast: None,
                error: Some(format!("Failed to read file: {}", e)),
            };
        }
    };
    
    let tree = parser.parse(&source_code, None).unwrap();
    let mut nodes = Vec::new();
    let mut node_counter = 0;
    
    walk_tree(tree.root_node(), &mut nodes, &mut node_counter);

    Response {
        id: request.id,
        path: request.path,
        ast: Some(nodes),
        error: None,
    }
}

// Recursive function to walk the AST and generate stable IDs
fn walk_tree(node: Node, nodes: &mut Vec<AstNode>, node_counter: &mut usize) -> usize {
    let current_id = *node_counter;
    *node_counter += 1;
    
    let mut child_ids = Vec::new();
    for i in 0..node.child_count() {
        let child = node.child(i).unwrap();
        child_ids.push(walk_tree(child, nodes, node_counter));
    }

    let start = node.start_position();
    let end = node.end_position();

    nodes.push(AstNode {
        id: current_id,
        kind: node.kind().to_string(),
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        start_point: (start.row, start.column),
        end_point: (end.row, end.column),
        children: child_ids,
    });
    
    current_id
}