"""this script contains helper functions to write a pbtx file for class labels"""

import csv
import os


def load_categories_from_csv_file(dataset):

    """Loads categories from a csv file.
    The CSV file should have one comma delimited numeric category id and string
    category name pair per line. For example:
    0,"cat"
    1,"dog"
    2,"bird"
    ...
    Args:
      dataset: path to dataset csv file
    Returns:
      categories: A list of dictionaries representing all possible categories.
                   The categories will contain an integer 'id' field and a string
                  'name' field.
    Raises:
      ValueError: If the csv file is incorrectly formatted.
    """
    categories = []

    with open(dataset, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            if not row:
                continue

            if len(row) != 3:
                raise ValueError('Expected 3 fields per row in csv: %s' % ','.join(row))

            if i > 0:
                category_id = int(i-1)
                category_display_name = row[1]
                categories.append({'class_id': category_id,
                                   'class_name': category_display_name,
                                   'display_name': category_display_name})
    return categories


def write_class_names(class_file):

    """saves the dict to .pbtxt file

    """
    # load the categories
    categories = load_categories_from_csv_file(class_file)

    # make pbtxt file
    categories.sort(key=lambda x: x['class_id'])
    with open(os.path.join(os.path.dirname(class_file), 'label_map.pbtxt'), 'w+') as f:
        count_sorted = [(cat['class_name'], cat['display_name']) for cat in categories]
        for i, class_name in enumerate(count_sorted):
            f.write("item {\n")
            f.write("    class_id: {}\n".format(i))
            f.write("    class_name: \'{}\'\n".format(class_name[0]))
            f.write("    display_name: \'{}\'\n".format(class_name[1]))
            f.write("}\n\n")
    return None


# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--label_map_csv_path',
#                         type=str,
#                         help='label map csv to read',
#                         default='/media/jay/data/Dataset/Hardware-Detection/class_names.csv',
#                         required=False)
#     parser.add_argument('--label_map_file',
#                         type=str,
#                         help='label map file to write',
#                         default='/media/jay/data/Dataset/Hardware-Detection/label_map.pbtxt',
#                         required=False)
#
#     return parser.parse_args(argv)
#
#
# if __name__ == '__main__':
#     write_class_names(parse_arguments(sys.argv[1:]))