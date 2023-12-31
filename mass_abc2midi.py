import sys, re
from subprocess import call

def main():
	abc_file = sys.argv[1]
	abc_file = open(abc_file, "r").read().split("\n\n")
	
	for i in range(len(abc_file)):
		if(len(abc_file[i]) > 2):
			f = open("test.abc", "w")
			f.write(abc_file[i])
			f.close()
	
			song_name = re.findall("T:(.*)", abc_file[i])[0].replace(" ", "_").replace("'","")
			song_key = re.findall("\nK:(.*)", abc_file[i])[0]
			filename = "midi_files/" + song_name + ':Key_' + song_key

			call(["./abc2midi.exe", "test.abc", "-o", filename])

main()
