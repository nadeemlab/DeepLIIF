import time
import os

'''
    ELog class define the path and content of train and test log.
'''

class ELog(object):

    def __init__(self,subId,f):
        if "TRAINING_ID" in os.environ:
            self.trainingId = os.environ["TRAINING_ID"]
        elif "DLI_EXECID" in os.environ:
            self.trainingId = os.environ["DLI_EXECID"]
        else:
            self.trainingId = ""
        self.subId = subId
        self.f = f

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def open(subId=None):
        if "LOG_DIR" in os.environ:
            folder = os.environ["LOG_DIR"]
        elif "JOB_STATE_DIR" in os.environ:
            folder = os.path.join(os.environ["JOB_STATE_DIR"],"logs")
        else:
            folder = "/tmp"

        if subId is not None:
            folder = os.path.join(folder, subId)

        if not os.path.exists(folder):
            os.makedirs(folder)

        f = open(os.path.join(folder, "stdout"), "a")
        return ELog(subId,f)

    def recordText(self,text):
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        timestr = "["+ timestr + "]"
        if self.f:
            self.f.write(timestr + " " + text + "\n")
            self.f.flush()

    def recordTrain(self,title,iteration,global_steps,loss,accuracy,worker):
        text = title
        text = text + ",	Timestamp: " + str(int(round(time.time() * 1000)))
        text = text + ",	Global steps: " + str(global_steps)
        text = text + ",	Iteration: " + str(iteration)
        text = text + ",	Loss: " + str(float('%.5f' % loss) )
        text = text + ",	Accuracy: " + str(float('%.5f' % accuracy) )
        self.recordText(text)

    def recordTest(self,title,loss,accuracy,worker):
        text = title
        text = text + ",	Timestamp: " + str(int(round(time.time() * 1000)))
        text = text + ",	Loss: " + str(float('%.5f' % loss) )
        text = text + ",	Accuracy: " + str(float('%.5f' % accuracy) )
        self.recordText(text)

    def close(self):
        if self.f:
            self.f.close()
