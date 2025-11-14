# Source Tree Server

This is a MCP (Machine Communication Protocol) server for analyzing and managing a source tree. The server provides various tools to retrieve information about files, directories, code metrics, and git repositories.

## Features

- **File Management**: Retrieve file names, file info, file content, and directory listings.
- **Language Detection**: Detect and count the number of files in different programming languages.
- **Git Integration**: Retrieve repository information, commit history, diffs, and search commits by pattern.
- **Code Metrics**: Analyze code metrics such as total lines of code, average lines per file, and cyclomatic complexity.

## Dependencies

The server requires the Python packages defined by `requirements.txt`.

## Usage

To start the server, run the following command:

```bash
python server.py --base-dir <path_to_source_tree> --mcp-host <host_name> --mcp-port <port_number> --transport <stdio|sse>
```

- `--base-dir`: The full qualified path to a source code directory.
- `--mcp-host`: Host name to listen on (default: 127.0.0.1).
- `--mcp-port`: Port to listen on (default: 8082).
- `--transport`: Which protocol to use (default: stdio).

## Tools

The server provides the following tools:

- **get_files**: Retrieve a list of file names in a directory.
- **get_file_info**: Retrieve information about a file.
- **get_file_content**: Retrieve the content of a file.
- **get_directories**: Retrieve a list of directory names in a directory.
- **get_languages**: Retrieve a dictionary of languages used inside the source tree.
- **get_repo_info**: Retrieve information about the git repository.
- **get_code_metrics**: Retrieve code metrics for the source tree.
- **get_line_counts**: Retrieve line counts for files.
- **get_last_n_commits**: Retrieve the last N commit hashes.
- **get_diff_for_commit**: Retrieve the diff between two commits.
- **search_commits_containing_change**: Search commits containing changes matching a pattern.

## Logging

The server logs information to `server.log` if it has write access to the current working directory. Otherwise, it logs to the console.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your improvements.