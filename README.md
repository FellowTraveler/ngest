# ngest
Python script for ingesting various files into a semantic graph. For text, images, cpp, python, rust, javascript, and PDFs.

```brew install llvm libmagic```

I had to add this in the imports:

```
    Config.set_library_path("/opt/homebrew/opt/llvm/lib")
    import clang.cindex
```

### Note

This project isn't done yet so don't bother using it yet.
When it's ready there will also be a Retriever script.
