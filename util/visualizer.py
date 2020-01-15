from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import os
from . import util, html
import numpy as np


class Serv(BaseHTTPRequestHandler):

	def do_GET(self):
		if self.path == '/':
			self.path = '/test.html'
		try:
			file_to_open = open(self.path[1:]).read()
			self.send_response(200)
		except:
			file_to_open = "File not found"
			self.send_response(404)
		self.end_headers()
		self.wfile.write(bytes(file_to_open, 'utf-8'))

	def log_message(self, format, *args):
		return

import matplotlib.pyplot as plt

class Visualizer():
	def __init__(self, opt):
		self.opt = opt
		self.display_id = opt.display_id
		self.name = opt.name

		self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
		self.img_dir = os.path.join(self.web_dir, 'images')
		print('create web directory %s...' % self.web_dir)
		util.mkdirs([self.web_dir, self.img_dir])

		if self.display_id > 0: 
			httpd = HTTPServer((opt.display_server,opt.display_port),Serv)
			threading.Thread(target=httpd.serve_forever, daemon=True).start()

		def get_concat_h(im1, im2):
		    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
		    dst.paste(im1, (0, 0))
		    dst.paste(im2, (im1.width, 0))
		    return dst

	def display_current_results(self, visuals, epoch, save_result):

		if self.display_id > 0:

			for label, image in visuals.items():
				aux_image = np.array(image[0] * 0.5 + 0.5)

			webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
