import argparse
from collections import defaultdict
import mimetypes
import datetime
import json
import os
import time
import uuid
import gradio as gr
import requests
from typing import Union
from PIL import Image
import cv2
import re

from serve.serving_utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
from serve.gradio_patch import Chatbot as grChatbot
from serve.gradio_css import code_highlight_css
from constants import LOGDIR
from conversation import get_conv_template, SeparatorStyle

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_DEMO_END_TOKEN = "<|endofchunk|>"
DEFAULT_ANSWER_TOKEN = "<answer>"
template_name = "drivegpt"

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Open Flamingo Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "otter_video": "aaaaaaa",
    "otter": "aaaaaab",
    "open_flamingo": "aaaaaac",
}


def extract_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "/home/yingzi/Visual-Instruction-Meta-Learning/serve/examples/temp_video.mp4"
        if "://" not in url:  # Local file
            video_path = url
        else:  # Remote URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (
        state,
        gr.Dropdown.update(choices=models, value=models[0] if len(models) > 0 else ""),
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot()) + ("", "", None,) * 2 + ("", None) * 1 + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, []) + ("", "", None,) * 2 + ("", None) * 1 + (disable_btn,) * 5


def add_text(
    state,
    model_selector,
    text_demo_question_1,
    text_demo_answer_1,
    video_demo_1,
    text_demo_question_2,
    text_demo_answer_2,
    video_demo_2,
    text_demo,
    video_demo,
    request: gr.Request,
):
    if text_demo_question_1 != "":
        text_demo_question_1 = text_demo_question_1.strip()
        if not re.search(r"[.,?]$", text_demo_question_1):
            text_demo_question_1 += "."
    if text_demo_answer_2 != "":
        text_demo_question_2 = text_demo_question_2.strip()
        if not re.search(r"[.,?]$", text_demo_question_1):
            text_demo_question_1 += "."
    if text_demo != "":
        text_demo = text_demo.strip()
        if not re.search(r"[.,?]$", text_demo):
            text_demo += "."
    template_name = "drivegpt"
    # print("++++++++++++++++++++++++++++++")
    # print(model_selector)

    text = text_demo
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if state is None:
        state =  get_conv_template(template_name).copy()
        logger.info(f"TEMPLATE. {state}")
    if len(text) <= 0 and video_demo is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot()) + ("", "", None,) * 2 +  ("", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot()) + ("", "", None,) * 2 + (moderation_msg, None) + (disable_btn,) * 5


    DEFAULT_ANSWER_TOKEN = "<answer> "
    human_role_label = state.copy().roles[0] + ": "
    bot_role_label = " " + state.copy().roles[1] + ":"

    is_video = True
    text = text[:1536]  # Hard cut-off
    if video_demo is not None:
        video_demo = get_image(video_demo)
        if isinstance(video_demo, list): is_video = True
        else: is_video = False

    if video_demo is not None and state is not None:
        state = get_conv_template(template_name).copy()
        logger.info(f"TEMPLATE. {state}")

    state.image_mark = "<image> is a driving video. " if is_video is True else "<image> is an image. "
    IMAGE_TOKEN = state.image_mark

    text = human_role_label + IMAGE_TOKEN + text

    if text_demo_answer_2 != "":
        assert video_demo_2 is not None
        video_demo_2 = get_image(video_demo_2)
        text = (
                human_role_label
                + IMAGE_TOKEN
                + text_demo_question_2
                + bot_role_label + " "
                # + DEFAULT_ANSWER_TOKEN
                + text_demo_answer_2
                + DEFAULT_DEMO_END_TOKEN
                + text
        )

    if text_demo_answer_1 != "":
        assert video_demo_1 is not None
        video_demo_1 = get_image(video_demo_1)
        text = (
                human_role_label
                + IMAGE_TOKEN
                + text_demo_question_1
                + bot_role_label + " "
                # + DEFAULT_ANSWER_TOKEN
                + text_demo_answer_1
                + DEFAULT_DEMO_END_TOKEN
                + text
        )

    print("="*50)
    print(text)

    input = (text, video_demo_1, video_demo_2, video_demo)
    state.append_message(state.roles[0], input)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot()) + ("", "", None,) * 2 + ("", None) * 1 + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(
    state,
    model_selector,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    early_stopping,
    request: gr.Request,
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    template_name = "drivegpt"

    if state.skip_next:
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        new_state = get_conv_template(template_name).copy()
        new_state.image_mark = state.image_mark
        new_state.conv_id = uuid.uuid4().hex
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn
        return

    # Construct prompt
    prompt = state.get_interleaved_prompt()
    prompt = prompt.strip()

    print(prompt)
    # if state.roles[1] is not None:
    #     role_label = state.roles[1] + ": "
    #     # hard code preprocessing: remove the last role label
    #     prompt = prompt[: -len(role_label)]

    # Construct generation kwargs
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "length_penalty": length_penalty,
        "do_sample": do_sample,
        "early_stopping": early_stopping,
    }

    # Make requests

    pload = {
        "model": model_name,
        "prompt": prompt,
        "stop": state.sep if state.sep_style == SeparatorStyle.ADD_COLON_SINGLE else state.sep2,
        "images": f"List of {len(state.get_images())} images",
        "generation_kwargs": generation_kwargs,
    }
    logger.info(f"==== request ====\n{pload}")

    pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=25,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    # output = data["text"][len(prompt) + 1 :].strip() # original postprocessing
                    output = data["text"].strip()  # TODO: fix hardcode postprocessing
                    # output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


title_markdown = ("""
# 🐬  Dolphins (the first version)
[[Project Page]](https://github.com/) [[Paper]](https://github.com/) [[Code]](https://github.com/) [[Model]](https://github.com/)
""")

tos_markdown = """
### Terms of Use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""


learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA. Please contact us if you find any potential violation.
"""


css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""
)


def build_demo(embed_mode):
    with gr.Blocks(title="DriveGPT Chat", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):

                with gr.Row():
                    with gr.Column():
                        videobox_1 = gr.Video(label="video 1 (optional)")

                    with gr.Column():
                        textbox_instruction_1 = gr.Textbox(
                            label="Instruction 1 (optional)",
                            show_label=True,
                            placeholder="Example: Describe this video.",
                        ).style(container=True)

                        textbox_answer_1 = gr.Textbox(
                            label="Answer 1 (optional)",
                            show_label=True,
                            placeholder="<Describe Demo Video 1>",
                        ).style(container=True)


            with gr.Column(scale=3):

                with gr.Row():
                    with gr.Column():
                        videobox_2 = gr.Video(label="Video 2 (optional)")

                    with gr.Column():
                        textbox_instruction_2 = gr.Textbox(
                            label="Instruction 2 (optional)",
                            show_label=True,
                            placeholder="Example: Describe this video.",
                        ).style(container=True)

                        textbox_answer_2 = gr.Textbox(
                            label="Answer 2 (optional)",
                            show_label=True,
                            placeholder="<Describe Demo Video 2>",
                        ).style(container=True)

        with gr.Row():
            with gr.Column(scale=3):
                model_selector = gr.Dropdown(choices=models, value=models[0] if len(models) > 0 else "", interactive=True, show_label=False).style(
                    container=False
                )

                videobox_3 = gr.Video(label="Video")

                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    max_new_tokens = gr.Slider(minimum=16, maximum=1500, value=512, step=1, interactive=True, label="# generation tokens")
                    temperature = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, interactive=True, label="temperature")
                    top_k = gr.Slider(minimum=0, maximum=10, value=0, step=1, interactive=False, label="top_k")
                    top_p = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, interactive=True, label="top_p")
                    no_repeat_ngram_size = gr.Slider(minimum=1, maximum=10, value=3, step=1, interactive=True, label="no_repeat_ngram_size")
                    length_penalty = gr.Slider(minimum=1, maximum=5, value=1, step=0.1, interactive=True, label="length_penalty")
                    do_sample = gr.Checkbox(value=True, interactive=True, label="do_sample")
                    early_stopping = gr.Checkbox(value=True, interactive=True, label="early_stopping")

            with gr.Column(scale=6):
                chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=500)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox_3 = gr.Textbox(
                            label="Text Query",
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                        ).style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        # cur_dir = os.path.dirname(os.path.abspath(__file__))
        # gr.Examples(
        #     examples=[
        #         ["", "", "", "", f"{cur_dir}/examples/Apple Vision Pro - Reveal Trailer.mp4", "Hey Otter, do you think it's cool? "],
        #         ["", "", "", "", f"{cur_dir}/examples/example.mp4", "What does the video describe?"],
        #         [
        #             "Is there a person in this video?",
        #             "Yes, a woman.",
        #             "",
        #             "",
        #             f"{cur_dir}/examples/dc_demo.mp4",
        #             "What does the video describe?",
        #         ],
        #         [
        #             "Is there a man in this video?",
        #             "Yes, he is riding a horse.",
        #             "What are the transports in this video?",
        #             "Tram, cars, and horse.",
        #             f"{cur_dir}/examples/dc_demo2.mp4",
        #             "What does the video describe?",
        #         ],
        #     ],
        #     inputs=[
        #         textbox_demo_question_1,
        #         textbox_demo_answer_1,
        #         textbox_demo_question_2,
        #         textbox_demo_answer_2,
        #         videobox_3,
        #         textbox_3,
        #     ],
        # )

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

        demo_list = [
            textbox_instruction_1, textbox_answer_1, videobox_1,
            textbox_instruction_2, textbox_answer_2, videobox_2,
        ]

        prarameter_list = [
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            no_repeat_ngram_size,
            length_penalty,
            do_sample,
            early_stopping,
        ]

        feedback_args = [textbox_3, upvote_btn, downvote_btn, flag_btn]

        upvote_btn.click(upvote_last_response, [state, model_selector], feedback_args)
        downvote_btn.click(downvote_last_response, [state, model_selector], feedback_args)
        flag_btn.click(flag_last_response, [state, model_selector], feedback_args)

        common_args = [state, chatbot] + demo_list + [textbox_3, videobox_3] + btn_list
        regenerate_btn.click(regenerate, state, common_args).then(http_bot, [state, model_selector] + prarameter_list, [state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, common_args)

        textbox_3.submit(add_text, [state, model_selector] + demo_list + [textbox_3, videobox_3], common_args).then(
            http_bot, [state, model_selector] + prarameter_list, [state, chatbot] + btn_list
        )
        submit_btn.click(add_text, [state, model_selector] + demo_list + [textbox_3, videobox_3], common_args).then(
            http_bot, [state, model_selector] + prarameter_list, [state, chatbot] + btn_list
        )

        widget_list = [state, model_selector, chatbot, textbox_3, submit_btn, button_row, parameter_row]

        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params], widget_list, _js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, widget_list)
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    gr.close_all()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default="7861")
    parser.add_argument("--controller_url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency_count", type=int, default=16)
    parser.add_argument("--model_list_mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    models = get_model_list()
    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )
    gr.close_all()