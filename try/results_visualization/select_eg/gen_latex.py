# -*- coding: utf-8 -*-

"""Generate latex table."""

import logging
import json
import os

logger = logging.getLogger(__name__)

result_file = "selected.json"
head_latex = r"""
%\begin{landscape}
\begin{table*}[]
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}cccccc@{}}
\toprule
\Large \textbf{Query} & \Large  \textbf{Method} & \Large \textbf{Demo 1} & \Large \textbf{Demo 2} & \Large \textbf{Demo 3} & \Large \textbf{Model Generation} \\ \midrule
"""

end_latex = r"""
\end{tabular}}
\end{table*}
%\end{landscape}
"""

template_comparison_latex = r"""
\multirow{2}{*}{\makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<QUERY_IMG_FILE>} \vspace{-0.0cm} \\ <QUERY_TEXT>}} 
& \Large MMICES &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<MMICES_DEMO_1_IMG_FILE>} \vspace{-0.0cm} \\ <MMICES_DEMO_1_TEXT>}  &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<MMICES_DEMO_2_IMG_FILE>} \vspace{-0.0cm} \\ <MMICES_DEMO_2_TEXT>} &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<MMICES_DEMO_3_IMG_FILE>} \vspace{-0.0cm} \\ <MMICES_DEMO_3_TEXT>} & \color{green} <MMICES_GENERATION> \\ \cmidrule(l){2-6} 
& \Large RICES &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<RICES_DEMO_1_IMG_FILE>} \vspace{-0.0cm} \\ <RICES_DEMO_1_TEXT>}  &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<RICES_DEMO_2_IMG_FILE>} \vspace{-0.0cm} \\ <RICES_DEMO_2_TEXT>} &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<RICES_DEMO_3_IMG_FILE>} \vspace{-0.0cm} \\ <RICES_DEMO_3_TEXT>} & \color{red} <RICES_GENERATION>  \\ \midrule
"""

# &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<MMICES_DEMO_4_IMG_FILE>} \vspace{-0.0cm} \\ <MMICES_DEMO_4_TEXT>
# &  \makecell{\includegraphics[height=3cm, width=4cm]{figures/coco/<RICES_DEMO_4_IMG_FILE>} \vspace{-0.0cm} \\ <RICES_DEMO_4_TEXT>

def extract_ques_and_ans(input_str):
    content = input_str.split("<image>")[1]
    content = content.split("<|endofchunk|>")[0]
    question = content.split("Short answer:")[0].strip().replace("Question:", "").strip()
    answer = content.split("Short answer:")[1].strip()
    return f"\\Large {question} \\\\ \\Large {answer}"


def download_coco_figure(img_file):
    if "val" in img_file:
        source_link = f"http://images.cocodataset.org/val2014/{img_file}"
    elif "train" in img_file:
        source_link = f"http://images.cocodataset.org/train2014/{img_file}"

    if not os.path.exists(f"figures/{img_file}"):
        cmd = f"wget {source_link} -O figures/{img_file}"
        os.system(cmd)

def extract_img_file(input_str):
    content = input_str.split("<image>")[0].strip()
    return content


results = json.load(open(result_file, "r"))

if os.path.exists("comparison.tex"):
    os.remove("comparison.tex")
with open("comparison.tex", "w") as f:
    f.write(" ")

all_latex = ""
fig_list = []
for count, que_id in enumerate(results):
    print(f"Processing {count}th question: {que_id}")
    query_img_file = results[que_id]["test_image"]
    print(query_img_file)
    query_text = results[que_id]["test_question"]
    mmices_generation = results[que_id]["prediction_okvqa_rice_image_ranking_text"]
    mmices_demos = results[que_id]["demo_okvqa_rice_image_ranking_text"][:4]
    rices_generation = results[que_id]["prediction_okvqa_rice_image"]
    rices_demos = results[que_id]["demo_okvqa_rice_image"][-4:]
    mmices_demo_1_img_file = extract_img_file(mmices_demos[0])
    mmices_demo_1_text = extract_ques_and_ans(mmices_demos[0])
    mmices_demo_2_img_file = extract_img_file(mmices_demos[1])
    mmices_demo_2_text = extract_ques_and_ans(mmices_demos[1])
    mmices_demo_3_img_file = extract_img_file(mmices_demos[2])
    mmices_demo_3_text = extract_ques_and_ans(mmices_demos[2])
    mmices_demo_4_img_file = extract_img_file(mmices_demos[3])
    mmices_demo_4_text = extract_ques_and_ans(mmices_demos[3])
    rices_demo_1_img_file = extract_img_file(rices_demos[0])
    rices_demo_1_text = extract_ques_and_ans(rices_demos[0])
    rices_demo_2_img_file = extract_img_file(rices_demos[1])
    rices_demo_2_text = extract_ques_and_ans(rices_demos[1])
    rices_demo_3_img_file = extract_img_file(rices_demos[2])
    rices_demo_3_text = extract_ques_and_ans(rices_demos[2])
    rices_demo_4_img_file = extract_img_file(rices_demos[3])
    rices_demo_4_text = extract_ques_and_ans(rices_demos[3])

    one_comparison_latex = template_comparison_latex.replace("<QUERY_IMG_FILE>", query_img_file)
    one_comparison_latex = one_comparison_latex.replace("<QUERY_TEXT>", f"\Large {query_text}")
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_1_IMG_FILE>", mmices_demo_1_img_file)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_1_TEXT>", mmices_demo_1_text)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_2_IMG_FILE>", mmices_demo_2_img_file)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_2_TEXT>", mmices_demo_2_text)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_3_IMG_FILE>", mmices_demo_3_img_file)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_3_TEXT>", mmices_demo_3_text)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_4_IMG_FILE>", mmices_demo_4_img_file)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_DEMO_4_TEXT>", mmices_demo_4_text)
    one_comparison_latex = one_comparison_latex.replace("<MMICES_GENERATION>", "\Large "+mmices_generation)

    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_1_IMG_FILE>", rices_demo_1_img_file)
    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_1_TEXT>", rices_demo_1_text)
    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_2_IMG_FILE>", rices_demo_2_img_file)
    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_2_TEXT>", rices_demo_2_text)
    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_3_IMG_FILE>", rices_demo_3_img_file)
    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_3_TEXT>", rices_demo_3_text)
    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_4_IMG_FILE>", rices_demo_4_img_file)
    one_comparison_latex = one_comparison_latex.replace("<RICES_DEMO_4_TEXT>", rices_demo_4_text)
    one_comparison_latex = one_comparison_latex.replace("<RICES_GENERATION>", "\Large "+rices_generation)

    fig_list.append(query_img_file)
    fig_list.append(mmices_demo_1_img_file)
    fig_list.append(mmices_demo_2_img_file)
    fig_list.append(mmices_demo_3_img_file)
    fig_list.append(mmices_demo_4_img_file)
    fig_list.append(rices_demo_1_img_file)
    fig_list.append(rices_demo_2_img_file)
    fig_list.append(rices_demo_3_img_file)
    fig_list.append(rices_demo_4_img_file)

    for fig in fig_list:
        download_coco_figure(fig)

    all_latex += one_comparison_latex
    all_latex += "\n"
    if (count+1) % 4 == 0:
        with open("comparison.tex", "a") as f:
            final = f"{head_latex}\n{all_latex}\n{end_latex}"
            f.write(final)
            print(f"write {count} to comparison.tex")
        all_latex = ""
    if count > 15:
        break
