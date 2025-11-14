import os
import logging
import argparse
import mimetypes
import datetime
import lizard
import pandas as pd

from pygount import SourceAnalysis, ProjectSummary
from git import Repo
from pathlib import Path
from collections import Counter
from fastmcp import FastMCP
from typing import Dict



logger = logging.getLogger(__name__)

mcp = FastMCP("Source Tree Server")

basedir = ""

@mcp.tool()
def get_files(folder_path:str = "", file_extension:str = "") -> list[str]:
    """Retrieve a list of file names in *folder_path*, which is, if provided, a directory inside
    the global runtime parameter basedir provided at start of server, that have the specified
    *file_extension*.

    Parameters
    ----------
    folder_path : str
        Optional path to the sub directory in basedir from which files should be listed.
        If an empty string is passed, the basedir is used as path. If a non empty string
        is passed, it will be joined to basedir global parameter.
    file_extension : str
        Optional file extension filter (e.g., ``".txt"``). Files must end with this
        extension to be included in the result. If an empty string is passed,
        all files are returned.

    Returns
    -------
    list[str]
        A list of file names located directly under *folder_path*, which exists as sub directory
        of runtime argument base-dir, that match the given extension. The list does not include directory paths.
    """
    logger.info(f"About to retrieve files from {folder_path}")
    
    entries = []
    
    global basedir
    
    dir = os.path.join(basedir, folder_path)
    
    if os.path.exists(dir):
        entries = os.listdir(dir)
        
    files = [f for f in entries if os.path.isfile(os.path.join(dir, f))]
    
    if len(file_extension) > 0:
        files = [f for f in files if f.endswith(file_extension)]
    
    return files
    

@mcp.tool()    
def get_file_info(path:str) -> Dict[str, str]:
    """Retrieve information about the file located at *path*, which is a directory inside
    the global runtime parameter basedir provided at start of server. The returned dictionary will include the following keys:

    - `name`: The name of the file.
    - `size`: The size of the file in bytes.
    - `modified_time`: The timestamp when the file was last modified, in ISO format (YYYY-MM-DDTHH:MM:SS).

    Parameters
    ----------
    path : str
        Path to the sub directory in basedir from which information should be retrieved. If a non empty string
        is passed, it will be joined to basedir global parameter.

    Returns
    -------
    Dict[str, str]
        A dictionary containing information about the file located at the given *path*. The keys and values are strings.
    """
    logger.info(f"About to retrieve file info from {path}")

    global basedir

    fullPath = os.path.join(basedir, path)

    if not os.path.isfile(fullPath):
        return {}

    mimeType, _ = mimetypes.guess_type(fullPath)

    stats = os.stat(fullPath)
    info = {
        'name': os.path.basename(fullPath),
        'size': str(stats.st_size),
        'modified_time': datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
        'type': str(mimeType)
    }

    return info
    
@mcp.tool()
def get_file_content(path:str) -> str:
    """Retrieve the content of the file located at *path*, which is a directory inside
    the global runtime parameter basedir provided at start of server. The returned string will contain the contents of the file.

    Parameters
    ----------
    path : str
        Path to the sub directory in basedir from which information should be retrieved. If a non empty string
        is passed, it will be joined to basedir global parameter.

    Returns
    -------
    str
        A string containing the contents of the file located at the given *path*.
    """
    logger.info(f"About to retrieve file content from {path}")

    global basedir

    fullPath = os.path.join(basedir, path)

    if not os.path.isfile(fullPath):
        return ""

    return Path(fullPath).read_text()

@mcp.tool()
def get_directories(path:str = "") -> list[str]:
    """Retrieve a list of directory names in *path*, which is, if provided, a subdirectory inside
    the global runtime parameter basedir provided at start of server.

    Parameters
    ----------
    path : str
        Optional path to the sub directory in basedir from which directories should be listed.
        If an empty string is passed, the basedir is used as path. If a non-empty string
        is passed, it will be joined to basedir global parameter.

    Returns
    -------
    list[str]
        A list of directory names located directly under *path*, which exists as a subdirectory
        of runtime argument base-dir.
    """
    logger.info(f"About to retrieve directories from {path}")

    fullPath = os.path.join(basedir, path)
    
    if not os.path.isdir(fullPath):
        return []
    
    return [d for d in os.listdir(fullPath) if os.path.isdir(os.path.join(fullPath, d))]

def detect_languages(path: str, languageCounter: Counter[str]) -> Counter[str]:
    """Retrieve a list of directory names in *path*, which is, if provided, a subdirectory inside
    the global runtime parameter basedir provided at start of server.

    Parameters
    ----------
    path : str
        Optional path to the sub directory in basedir from which directories should be listed.
        If an empty string is passed, the basedir is used as path. If a non-empty string
        is passed, it will be joined to basedir global parameter.
    languageCounter : Counter[str]
        A counter object that keeps track of the number of files found for each language.

    Returns
    -------
    Counter[str]
        An updated counter object containing the number of files found for each language in the specified directory.
    """
    languageMap = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c' : 'C',
        '.cs' : 'C#',
        '.rb' : 'Ruby',
        '.php' : 'PHP',
        '.go' : 'Go',
        '.ts' : 'TypeScript',
        '.rs' : 'Rust',
        '.swift' : 'Swift'
    }
    
    numFiles = 0
    
    for root, _, files in os.walk(path): # type: ignore
        for file in files:
            if os.path.isdir(os.path.join(path, file)):
                detect_languages(os.path.join(path, file), languageCounter)
            else:
                _, ext = os.path.splitext(file)
                if ext in languageMap:
                    languageCounter[languageMap[ext]] += 1
                    numFiles = numFiles + 1
    
    return languageCounter


def get_language_stats() -> dict[str,str]:
    """Retrieve a dictionary of languages and the count of files in each language in the global runtime parameter basedir provided at start of server.

    Returns
    -------
    dict[str,str]
        A dictionary containing the number (as string) of files found for each language.
    """
    global basedir
    
    languageCounter:Counter[str] = Counter()
    
    languageCounter = detect_languages(basedir, languageCounter)
    
    languages:dict[str,str] = {}
    
    for lang, count in languageCounter.most_common():
        languages[lang] = str(count)
    
    return languages

def get_source_extensions() -> list[str]:
    """Retrieve a list of source file extensions found in the global runtime parameter basedir provided at start of server.

    Returns
    -------
    list[str]
        A list containing the unique source file extensions found in the specified directory.
    """
    languages = get_language_stats()
    
    ext:list[str] = []
    match(next(iter(languages.keys()))):
        case "Java":
            ext.append("java")
        case "C++":
            ext.append("cpp")
            ext.append("hpp")
            ext.append("c")
            ext.append("h")
        case "C":
            ext.append("c")
        case "PHP":
            ext.append("php")
        case "Python":
            ext.append("py")
        case _:
            pass
    return ext                        
    
@mcp.tool()
def get_languages() -> dict[str,str]:
    """Retrieve a dictionary of languages used inside the source tree in the global runtime parameter basedir provided at start of server.

    Returns
    -------
    dict[str,str]
        A dictionary containing the number (as string) of files found for each language.
    """
    return get_language_stats()


@mcp.tool()
def get_repo_info() -> dict[str,str]:
    """Retrieve information about the git repository located at the global runtime parameter basedir provided at start of server. The returned dictionary will include the following keys:

    - `currentBranch`: The name of the current branch
    - `lastCommit`: The hash of the last commit
    - `author`: The author of the last commit
    - `date`: The date and time of last commit
    - `message`: The commit message of the last commit
    - `branches`: A list of all branch names in the repository.
    - `last_five_commits`: A list of the last five commit hashes.

    Returns
    -------
    dict[str,str]
        A dictionary containing information about the git repository located at the given *basedir*. The keys and values are strings.
    """
    global basedir
    
    repoInfo:dict[str,str] = {}
    
    gitDir = os.path.join(basedir, '.git')
    
    if os.path.exists(gitDir) and os.path.isdir(gitDir):
        repo = Repo(basedir)
        current_branch = repo.head.ref.name
        repoInfo['currentBranch'] = current_branch
        commit = repo.head.commit
        repoInfo['lastCommit'] = commit.hexsha
        repoInfo['author'] = f"{commit.author.name} <{commit.author.email}>"
        repoInfo['date'] = commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
        repoInfo['message'] = str(commit.message)
        
        branches = ""
        for branch in repo.branches:
            branches = f"{branches}{branch.name}, "
        branches = branches[:-2] # remove
        
        repoInfo['branches'] = branches

        commits = ""
        for c in repo.iter_commits(current_branch, max_count=5):
            commits = f"{commits}{c.hexsha[:7]} - {c.author.name} - {c.summary}, "
        commits = commits[:-2] # remove
        
        repoInfo['last_five_commits'] = commits
    
    return repoInfo

def lizard_analysis_to_dataframe(results: list[lizard.FileInformation]) -> pd.DataFrame:
    """Convert a list of Lizard file analysis results to a pandas DataFrame.

    Parameters
    ----------
    results : list[lizard.FileInformation]
        A list of FileInformation objects returned by the Lizard tool.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the analysis results, with columns for each metric provided by Lizard.
    """
    records:list[dict[str,str]] = []
    
    for fileinfo in results:
        nloc_file = fileinfo.nloc
        max_ccn:int = 0
        
        for func_info in fileinfo.function_list:
            if int(func_info.cyclomatic_complexity) > max_ccn:
                max_ccn = int(func_info.cyclomatic_complexity)

        records.append({
            'filename': fileinfo.filename,
            'max_ccn': str(max_ccn),
            'nloc': nloc_file,
            'func_count': str(len(fileinfo.function_list))
        })
    return pd.DataFrame(records)

def get_files_for_extension(path:str, exts:list[str]) -> list[str]:
    """Retrieve a list of file names in *path*, which is, if provided, a directory inside
    the global runtime parameter basedir provided at start of server, that have any of the specified
    *exts*.

    Parameters
    ----------
    path : str
        Optional path to the sub directory in basedir from which files should be listed.
        If an empty string is passed, the basedir is used as path. If a non empty string
        is passed, it will be joined to basedir global parameter.
    exts : list[str]
        List of file extensions to filter by (e.g., ``[".txt", ".md"]``). Files must end with any of these
        extensions to be included in the result. If an empty list is passed,
        all files are returned.

    Returns
    -------
    list[str]
        A list of file names located directly under *path*, which exists as sub directory
        of runtime argument base-dir, that match the given extension(s). The list does not include directory paths.
    """
    logger.info(f"About to get files for extensions in path {path}")
    
    filesToAnalyze:list[str] = []
    
    for root, _, files in os.walk(path): # type: ignore
        for file in files:
            entry = os.path.join(root, file).replace("\\", "/")
            if entry.find("/.git/") > 0:
                continue
            
            logger.debug(f"About to check file: {entry}")
            
            if not os.path.isdir(entry):
                if file.endswith(tuple([f".{ext}" for ext in exts])):
                    logger.debug(f"Adding {entry} to list of files to analyze")
                    filesToAnalyze.append(entry)
    
    return filesToAnalyze

@mcp.tool()
def get_code_metrics() -> dict[str, str]:
    """Retrieve code metrics for the source tree in the global runtime parameter basedir provided at start of server.
    
    The returned dictionary will include the following keys:

    - `total_files`: The total number of files in the source tree.
    - `total_lines`: The total number of lines of code in the source tree.
    - `average_loc_per_file`: The average number of lines of code per file in the source tree.
    - `languages`: A dictionary containing the number (as string) of files found for each language.

    Returns
    -------
    dict[str, str]
        A dictionary containing code metrics for the source tree located at the given *basedir*. The keys and values are strings.
    """
    global basedir
    
    df = pd.DataFrame()
    filesToAnalyze = get_files_for_extension(basedir, get_source_extensions())
    logger.info(f"About to analyze {len(filesToAnalyze)} files")
    
    fi:list[lizard.FileInformation] = []
    
    for file in filesToAnalyze:    
        fi.append(lizard.analyze_file.analyze_source_code(file, Path(file).read_text()))

    df = lizard_analysis_to_dataframe(fi)
    return df.to_json()

@mcp.tool()
def get_line_counts() -> dict[str, int]:
    """Retrieve a dictionary of file names and their corresponding line counts in the global runtime parameter basedir provided at start of server.

    Returns
    -------
    dict[str, int]
        A dictionary containing the number (as integer) of lines for each file.
    """
    global basedir
    
    filesToAnalyze = get_files_for_extension(basedir, get_source_extensions())
    logger.info(f"About to analyze {len(filesToAnalyze)} files")
    
    projectSummary = ProjectSummary()
    
    for file in filesToAnalyze:
        projectSummary.add(SourceAnalysis.from_file(file, group="pygount"))

    return str(projectSummary)

@mcp.tool()
def get_last_n_commits(count:int = 10) -> list[str]:
    """Retrieve a list of the last *count* commit hashes in the global runtime parameter basedir provided at start of server.

    Parameters
    ----------
    count : int, optional
        The number of commits to retrieve. Defaults to 10.

    Returns
    -------
    list[str]
        A list containing the last *count* commit hashes.
    """
    global basedir

    commits:list[str] = []

    gitDir = os.path.join(basedir, '.git')
    
    if os.path.exists(gitDir) and os.path.isdir(gitDir):
        repo = Repo(basedir)
        
        for commit in repo.iter_commits(max_count=count):
            commits.append(str(commit))
    
    return commits

@mcp.tool()
def get_diff_for_commit(older_commit_hash: str = 'HEAD~1', newer_commit_hash: str = 'HEAD') -> str:
    """Retrieve the diff between two commits in the global runtime parameter basedir provided at start of server.

    Parameters
    ----------
    older_commit_hash : str, optional
        The hash of the older commit. Defaults to 'HEAD~1'.
    newer_commit_hash : str, optional
        The hash of the newer commit. Defaults to 'HEAD'.

    Returns
    -------
    str
        A string containing the diff between the two commits.
    """

    global basedir
    gitDir = os.path.join(basedir, '.git')
    
    if os.path.exists(gitDir) and os.path.isdir(gitDir):
        repo = Repo(basedir)
        diffs = repo.commit(older_commit_hash).diff(newer_commit_hash, create_patch=True)
        diff = ""
        for diffitem in diffs:
            diff = diff + str(diffitem)

        return diff
    
    return "No git repository found in the basedir."

@mcp.tool()
def search_commits_containing_change(pattern:str) -> str:
    """Retrieve a list of commit hashes that contain changes matching the given pattern in the global runtime parameter basedir provided at start of server.

    Parameters
    ----------
    pattern : str
        The pattern to search for in commit messages or diffs.

    Returns
    -------
    str
        A string containing the commit hashes separated by newlines.
    """
    global basedir
    
    
    commits:str = ""
    gitDir = os.path.join(basedir, '.git')
    if os.path.exists(gitDir) and os.path.isdir(gitDir):
        repo = Repo(basedir)
        commits = repo.git.log(G=pattern, pretty='oneline')

    return commits

def main():
    """Start the server and handle incoming requests. This method will initialize the global basedir variable with the value provided as runtime argument, then start the FastMCP server.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logging.basicConfig(
        filename="server.log",
        encoding="utf-8",
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description="MCP Server for a source tree")
    parser.add_argument("--base-dir", type=str, help="A full qualified path to a source code directory")
    parser.add_argument("--mcp-host", type=str, default="127.0.0.1", help="Host name to listen, default: 127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=8082, help="Port to listen on, default: 8082")
    parser.add_argument("--transport", type=str, default="stdio", choices=["stdio", "sse"],
                        help="Which protocol to use, default: stdio")
    

    args = parser.parse_args()
    
    global basedir
    basedir = args.base_dir
    
    if args.transport == "sse":
        logger.info(f"Starting Source Tree MCP server")
        mcp.settings.log_level = "INFO"
        mcp.settings.host = args.mcp_host
        mcp.settings.port = args.mcp_port
        
        mcp.run(transport="sse")
    else:
        mcp.run()

# Execute the main method if started as an application
if __name__ == "__main__":
    main()