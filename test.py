import sys
from subprocess import call

def main():
    abc_file = sys.argv[1]
    abc_file = open(abc_file, "r").read().split("\n\n")
    print(abc_file[0])
    print(abc_file[1])

    call(["./abc2midi.exe", abc_file[0]])

main()