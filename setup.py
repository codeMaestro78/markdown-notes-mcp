from setuptools import setup, find_packages

setup(
    name="markdown-notes-mcp",
    version="1.0.0",
    description="Markdown Notes MCP with Smart Collections",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "python-frontmatter>=1.0.0",
        "nltk>=3.8",
    ],
    python_requires=">=3.8",
)
