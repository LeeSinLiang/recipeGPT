from io import StringIO
from gpt import GPT
from config import GPTConfig
from utils import Tokenizer
from transformers import GPT2TokenizerFast
import json
import torch
import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title='microGPT: Recipe Generation!', page_icon='😋')

@st.cache_resource
def load_model(model_file, cfg):
	model = GPT(cfg.vocab_size, cfg.max_length, cfg.n_emb, cfg.n_head, cfg.n_layer)
	model = model.to(device)
	print('----')
	model.load_state_dict(torch.load(model_file, map_location=device))
	model.eval()

	return model

@st.cache_resource
def load_tokenizer(tokenizer_file, mode):
	if mode == 'bpe':
		tokenizer = GPT2TokenizerFast(tokenizer_file = tokenizer_file, model_max_length=256)		
	elif mode == 'char':
		tokenizer = Tokenizer(tokenizer_file=tokenizer_file)
	else:
		raise Exception('Incorrect mode.')
	return tokenizer

hide_menu_style = """
	<style>
	#MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
	</style>
	"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title('microGPT: Generate Recipe! 😋')

# print(decode(encode('Hello World')))
with open('config/config.json', 'r') as f:
    config = json.load(f)

cfg = GPTConfig(**config)

with st.spinner('Loading Model...'):
	model = load_model('models/recipeBPE-2.2876', cfg)
	tokenizer = load_tokenizer('config/tokenizer.json', 'bpe')

with st.form("input"):
	st.write('<p style="font-size: 20px;">Recipe Title:</p>', unsafe_allow_html=True)
	context = st.text_input(label='Input: ', placeholder="Text here...", label_visibility='collapsed')
	st.write('<p style="margin-bottom: 1%;font-size: 20px;">Number of characters to generate:</p>', unsafe_allow_html=True)
	n_tokens = st.slider(label='Number of tokens', help='How many characters do you want generate?', min_value=0, max_value=5000, value=500, step=50, label_visibility='collapsed')
	st.write('<p style="margin-bottom: 1%;font-size: 20px;">Top-K Sampling :</p>', unsafe_allow_html=True)
	top_k = st.slider(label='Top K', min_value=0, max_value=cfg.vocab_size, value=0, step=1, label_visibility='collapsed')
	st.write('<p style="margin-bottom: 1%;font-size: 20px;">Top-P Sampling (Nucleus Sampling) :</p>', unsafe_allow_html=True)
	top_p = st.slider(label='Top P', min_value=0.0, max_value=1.0, value=0.9, step=0.1, label_visibility='collapsed')
	st.write('<p style="margin-bottom: 1%;font-size: 20px;">Temperature :</p>', unsafe_allow_html=True)
	temperature = st.slider(label='Tempeature', min_value=0.0, max_value=1.0, value=0.8, step=0.1, label_visibility='collapsed')
	submitted = st.form_submit_button("Submit")

if submitted:
	with torch.no_grad():
		# print(tokenizer.decode(model.generate(context, max_tokens_generate=500).tolist()))
		context_str = 'Title: ' + context
		context = torch.tensor(tokenizer.encode(context_str), dtype=torch.long, device=device).reshape(1, -1)
		progress_text = "Operation in progress. Please wait."
		progress_bar = st.progress(0, text=progress_text)
		placeholder = st.empty()
		output_fn = getattr(placeholder, 'text')
		with StringIO() as buffer:
			def out_fn(b):
				buffer.write(b + '')
				output_fn(buffer.getvalue() + '')
			out_fn(context_str)
			response = tokenizer.decode(model.generate(tokenizer.decode, out_fn=out_fn, progress=(progress_bar, progress_text), idx=context, max_tokens_generate=n_tokens, temperature=temperature, top_k=top_k, top_p=top_p).tolist())
		st.balloons()

footer="""<style>
a:link , a:visited{
	color: rgb(250, 250, 250);
	text-decoration: none;
}

a:hover,  a:active {
	text-decoration: underline;
}

.footer {
	position: fixed;
	left: 0;
	bottom: 0;
	width: 100%;
	color: rgb(250, 250, 250);
	text-align: center;
	height: 6%;
}

.content {
	margin: 0;
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
}

</style>
<div class="footer"><div class="content">
<p style="color: rgba(250, 250, 250, 0.4)">Made and developed by <a href="https://www.github.com/LeeSinLiang">Lee Sin Liang</a></p>
</div></div>
"""
st.markdown(footer,unsafe_allow_html=True)
