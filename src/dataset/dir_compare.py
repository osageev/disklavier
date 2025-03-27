import os

paths = [
    "/media/nova/Datasets/sageev-midi/20250320/trimmed",
    "/media/nova/Datasets/sageev-midi/20250110/unsegmented"
]
# p_a = "/Users/finlay/Documents/Ableton Live/clean sessions/clean audio"
# p_m = "/Users/finlay/Documents/Ableton Live/clean sessions/clean midi"
# p_t = "/Users/finlay/Documents/Ableton Live/trimming Projects/trimmed"



def compare_directory_contents(directories):
    """
    Compare file lists across multiple directories and show differences.

    Parameters
    ----------
    directories : list
        List of paths to directories to compare.

    Returns
    -------
    None
        Prints the comparison results to stdout.
    """
    # get files from each directory and store in a dict
    dir_files = {}
    all_files = set()

    for directory in directories:
        if not os.path.exists(directory):
            print(f"warning: directory {directory} does not exist")
            continue

        # get all files and remove extensions
        files = []
        for f in os.listdir(directory):
            name = os.path.splitext(f)[0]
            files.append(name)
            all_files.add(name)
        dir_files[directory] = set(files)

    # for each file that exists anywhere, check where it's missing
    print("\nFile presence across directories:")
    for file in sorted(all_files):
        # find which directories have and don't have this file
        present_in = []
        missing_from = []
        for directory in dir_files:
            if file in dir_files[directory]:
                present_in.append(os.path.basename(directory))
            else:
                missing_from.append(os.path.basename(directory))

        # only show files that are missing from at least one directory
        if missing_from:
            print(f"\n{file}:")
            print(f"  present in: {', '.join(present_in)}")
            print(f"  missing from: {', '.join(missing_from)}")


compare_directory_contents(paths)
