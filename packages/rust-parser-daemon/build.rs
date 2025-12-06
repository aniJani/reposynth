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

    // CSS
    let dir_css: PathBuf = ["grammars", "tree-sitter-css", "src"].iter().collect();
    cc::Build::new()
        .include(&dir_css)
        .file(dir_css.join("parser.c"))
        .file(dir_css.join("scanner.c"))
        .compile("tree-sitter-css");

    // SCSS
    let dir_scss: PathBuf = ["grammars", "tree-sitter-scss", "src"].iter().collect();
    cc::Build::new()
        .include(&dir_scss)
        .file(dir_scss.join("parser.c"))
        .file(dir_scss.join("scanner.c"))
        .compile("tree-sitter-scss");

    // HTML
    let dir_html: PathBuf = ["grammars", "tree-sitter-html", "src"].iter().collect();
    cc::Build::new()
        .include(&dir_html)
        .file(dir_html.join("parser.c"))
        .file(dir_html.join("scanner.c"))
        .compile("tree-sitter-html");
}
