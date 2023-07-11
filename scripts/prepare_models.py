'''
Generates Models.md content based on csv data found in data folder.

Usage:
------
python3 prepare_models.py <input data folder> <target to folder to generate Models.md>

'''
import sys, logging, os, glob

# Some Constants
MODEL_FILE_NAME = "Models.md"
MODEL_TABLE_ENDING =  " |"
MODEL_FILE_FORMAT = """# {title}

{header_info}
{data}

"""

# Some Helper Functions

def extract_path_segments(path, sep=os.sep):
    path, filename = os.path.split(os.path.abspath(path))
    bottom, rest = path[1:].split(sep, 1)
    bottom = sep + bottom
    middle, top = os.path.split(rest)
    return (bottom, middle, top)

def process_csv_format(content):
    contents = content.split("\n")

    header_info = "| "
    data = ""

    # First line should be the cotent of file
    title = contents[0]

    # Header details
    header_count = 0
    link_mapping = {}
    for i,line in enumerate(contents[1].split(",")):
        if line and len(line.strip()) > 0:
            header_count += 1

            if "link" in line.lower():
                # i-1's link is provided by next element
                link_mapping[i-1] = i
            else:
                header_info += line.strip() + MODEL_TABLE_ENDING

    header_info += "\n| "
    for _ in range(header_count - len(link_mapping)):
        header_info += "--------- |"

    # Process Data
    for line in contents[2:]:
        data_line = "| "

        items = [item.strip() for item in line.split(",") if item and len(item.strip()) > 0 ]
        
        for i, item in enumerate(items):
            # Link is already processed
            if i in link_mapping.values(): continue

            # handle linking
            if i in link_mapping:
                item = item.strip()
                link = items[link_mapping[i]]
                target_link = "[" + item + "](" + link + ") "
                data_line += target_link + MODEL_TABLE_ENDING
            else:
                data_line += item.strip() + MODEL_TABLE_ENDING
        
        if len(data_line) > 2:
            data += data_line + "\n"

    return MODEL_FILE_FORMAT.format(
        title = title,
        header_info = header_info,
        data = data
    )


# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # need to update root level logger
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console.setFormatter(console_format)
logger.addHandler(console)

if len(sys.argv) != 3:
    logger.error("Missing parameter for input & target folder!")
    sys.exit(1)

DATA_PATH = sys.argv[1]
TARGET_PATH = sys.argv[2]

# Handle missing/invalid file path
if not os.path.exists(DATA_PATH):
    logger.error("Unknown folder path: %s", DATA_PATH)
    sys.exit(1)

# find all csv files inside data folder
csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"), recursive=True)

logger.debug("Found %s CSV files in '%s'", len(csv_files), DATA_PATH)

result = ""

for csv_file in csv_files:
    # Generate Models.md content
    result += process_csv_format(open(csv_file, 'r').read()) + "\n\n\n"

with open(os.path.join(TARGET_PATH, MODEL_FILE_NAME), 'w') as f:
    f.write(result)

logger.debug("Updated Models in %s", TARGET_PATH)
