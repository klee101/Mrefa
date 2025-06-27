import argparse

def convert_dos2unix(input_file, output_file):
    """
    Convert DOS linefeeds (CRLF) to Unix linefeeds (LF).
    """
    content = ''
    outsize = 0
    with open(input_file, 'rb') as infile:
        content = infile.read()
    with open(output_file, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + b'\n')
    print("Done. Saved %s bytes." % (len(content) - outsize))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DOS linefeeds (CRLF) to Unix linefeeds (LF).")
    parser.add_argument('-i', '--input', required=True, help="Path to the input file (source).")
    parser.add_argument('-o', '--output', required=True, help="Path to the output file (destination).")

    args = parser.parse_args()
    convert_dos2unix(args.input, args.output)