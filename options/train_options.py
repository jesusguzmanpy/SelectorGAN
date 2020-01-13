from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--display_freq', type=int, default=400, help='frecuencia para mostrar resultados en la pantalla')
        parser.add_argument('--display_ncols', type=int, default=4, help='si es positivo, ver todas las fotos, cantidad por filas')
        parser.add_argument('--display_id', type=int, default=1, help='id de la ventana para ver en la web')
        parser.add_argument('--no_html', action='store_true', help='no guardar resultados intermedios [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server para ver la web')
        parser.add_argument('--display_env', type=str, default='main', help='nombre del enviroment visdom')
        parser.add_argument('--display_port', type=int, default=8097, help='puerto del visdom para ver en la web')

        parser.add_argument('--epoch_count', type=int, default=1, help='el recuento de la época inicial, guardamos el modelo por <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        parser.add_argument('--n_epochs', type=int, default=100, help='Número de épocas con la tasa de aprendizaje inicial')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='número de épocas para decaer linealmente la tasa de aprendizaje a cero')
 
        self.isTrain = True
        return parser