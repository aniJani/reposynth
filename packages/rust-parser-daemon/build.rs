use std::path::PathBuf;

fn main() {
    let dir: PathBuf = ["grammars", "tree-sitter-python", "src"].iter().collect();
    cc::Build::new()
        .include(&dir)
        .file(dir.join("parser.c"))
        .file(dir.join("scanner.c"))
        .compile("tree-sitter-python");

    let dir_ts: PathBuf = ["grammars", "tree-sitter-typescript", "typescript", "src"]
        .iter()
        .collect();
    cc::Build::new()
        .include(&dir_ts)
        .file(dir_ts.join("parser.c"))
        .file(dir_ts.join("scanner.c"))
        .compile("tree-sitter-typescript");
}
