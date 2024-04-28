# -*- coding: utf-8 -*-

"""Remove unnecessary images in Submission folder."""

import logging
import os

logger = logging.getLogger(__name__)

base_dir = "/Users/shuochen/Downloads/CVPR-arXiv"

example_file = base_dir + "/supp/6-examples.tex"
fig_dir = base_dir + "/figures/coco"

# read the example file into a single string
with open(example_file, "r") as f:
    example_str = f.read()

# interate through the figures in the figure directory
total_fig = 0
removed_fig = 0
for fig in os.listdir(fig_dir):
    if fig.endswith(".jpg"):
        fig_name = fig
        total_fig += 1
        if fig_name not in example_str:
            os.remove(fig_dir + "/" + fig)
            print("Removed %s" % fig)
            removed_fig += 1

print("Total figures: %d" % total_fig)
print("Removed figures: %d" % removed_fig)