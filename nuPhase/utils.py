
def strip_file_extension(file_name: str, extension: str) -> str:

    out_file = str(file_name)

    if out_file.split(".")[-1] == extension:
        out_file = ".".join(out_file.split(".")[:-1])

    return out_file
