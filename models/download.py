from subprocess import Popen, PIPE

if __name__ == '__main__':
    filename = 'download.csv'
    with open(filename, 'r') as file:
        data = file.readlines()

    for line in data:
        folder = line.split(',')[0].strip()
        address = line.split(',')[1].strip()

        process = Popen(["wget", address, "-P", folder], stdout=PIPE)
        (output, err) = process.communicate()
        exit_code = process.wait()
        print(output)
