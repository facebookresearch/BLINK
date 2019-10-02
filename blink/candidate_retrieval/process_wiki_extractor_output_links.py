# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import xml.etree.ElementTree as ET
import io
import re
import argparse
import os
import pickle
import sys
import urllib.parse
import regex

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input", type=str, help="The full path to the file to process", required=True
)
parser.add_argument(
    "--output", type=str, help="The full path to the output file", required=True
)

args = parser.parse_args()


input_file_path = args.input
output_file_path = args.output

if not os.path.isfile(input_file_path):
    print("Input file `{}` doesn't exist!".format(output_file_path))
    sys.exit()

if os.path.isfile(output_file_path):
    print("Output file `{}` already exists!".format(output_file_path))
    sys.exit()

xml_end_tag = "</doc>"

entities_with_duplicate_titles = set()
title2id = {}

id_title2parsed_obj = {}

num_lines = 0
with io.open(input_file_path, mode="rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        num_lines += 1

c = 0
pattern = re.compile("(<a href=([^>]+)>((?:.(?!\<\/a\>))*.)<\/a>)")
docs_failed_xml = 0

with io.open(input_file_path, mode="rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        c += 1

        if c % 1000000 == 0:
            print("Processed: {:.2f}%".format(c * 100 / num_lines))

        if line.startswith("<doc id="):
            doc_xml = ET.fromstring("{}{}".format(line, xml_end_tag))
            doc_attr = doc_xml.attrib

            lines = []

        lines.append(line.strip())

        if line.startswith("</doc>"):
            temp_obj = {}
            temp_obj["url"] = doc_attr["url"]
            temp_obj["lines"] = lines

            text = " ".join([l for l in lines if l != ""])
            try:
                doc_xml = ET.fromstring(text)
                links_xml = doc_xml.getchildren()
                links = []

                for link_xml in links_xml:
                    link = {}
                    link["xml_attributes"] = link_xml.attrib
                    link["text"] = link_xml.text.strip()
                    link["href_unquoted"] = urllib.parse.unquote(
                        link_xml.attrib["href"]
                    )

                    link_xml.tail = ""
                    link["raw"] = ET.tostring(
                        link_xml, encoding="unicode", method="xml"
                    )
                    links.append(link)

                temp_obj["links_xml"] = links
            except Exception as e:
                temp_obj["links_xml"] = None
                docs_failed_xml += 1

            text = " ".join([l for l in lines[1:-1] if l != ""])
            links = []

            for match in pattern.findall(text):
                raw, href, text = match

                link = {}
                link["raw"] = raw
                link["href_unquoted"] = urllib.parse.unquote(href.strip('"'))
                link["text"] = text

                links.append(link)

            temp_obj["links_regex"] = links

            id_, title = doc_attr["id"], doc_attr["title"]

            key = (id_, title)
            id_title2parsed_obj[key] = temp_obj

            # # check for duplicate titles
            # if title in title2id:
            #     entities_with_duplicate_titles.add(id_)
            #     entities_with_duplicate_titles.add(title2id[title])
            #     print("DUPLICATE TITLE:", id_, title2id[title])
            # else:
            #     title2id[title] = id_

print("Processed: {:.2f}%".format(c * 100 / num_lines))
print("Dumping", output_file_path)
pickle.dump(id_title2parsed_obj, open(output_file_path, "wb"), protocol=4)
# print('Portion of documents with improper xml: {:.2f}%'.format(docs_failed_xml*100/len(id_title2parsed_obj)))
