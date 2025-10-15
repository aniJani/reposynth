cd packages/rust-parser-daemon
cargo build --release
cd ../..
python packages/python-orchestrator/orchestrator/parser_client.py