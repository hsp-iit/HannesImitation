import multiprocessing
import cv2


class HannesFrameCapture(multiprocessing.Process):
    """
    """

    def __init__(self, 
                 camera_index=0,
                 kwargs={},):
        
        super().__init__()
        
        self.camera_index = camera_index

        self.kwargs = kwargs
        self.flag_demo = kwargs['flag_demo']
        self.frames_list = kwargs['frames_list']
        self.lock = kwargs['lock']

        # open camera
        self.cam = cv2.VideoCapture(self.camera_index)

        if not self.cam.isOpened():
            print("Cannot open camera")

        # Set properties
        # TODO: make a method set_property or similar
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # default 3, 1 disables autoexposure
        self.cam.set(cv2.CAP_PROP_EXPOSURE, 500) # default 166. 300=30fps, 400=25fps, 500=20fps
        self.cam.set(cv2.CAP_PROP_GAIN, 100) # 0-128, default 64 (gain=100 is ok when auto exposure is disabled and exposure is 500)
        assert(self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1)
        assert(self.cam.get(cv2.CAP_PROP_EXPOSURE) == 500)
        assert(self.cam.get(cv2.CAP_PROP_GAIN) == 100)

        # if frame is read correctly ret is True
        ret, frame = self.cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")

    def run(self):
        
        while self.flag_demo.is_set():
            # read new frame   
            ret, image_cv = self.cam.read()

            # append image to list in mutual exclusion
            # NOTE: vedere se posso usarlo con una lock nulla per riutilizzare codice per telecamera esterna.
            with self.lock:
                self.frames_list.append(image_cv)

        # release camera at the end
        self.cam.release()
        print("camera released")