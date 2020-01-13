import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():

    def __init__(self, opt):

        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        
        if self.display_id > 0: #conección a visdom
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, use_incoming_socket=False)
            if not self.vis.check_connection():
                try:
                    self.create_visdom_connections()
                except Exception as e:
                    "print('\n\nNo se puede conectar a Visdom. \n Intentando inicializar Visdom....')"      

        if self.use_html:  # crear un HTML en <checkpoints_dir>/web/; las imágenes seran guardados en <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('crear dirección web %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

    def create_visdom_connections(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        #print('\n\nNo se puede conectar a Visdom. \n Intentando inicializar Visdom....')
        #print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)