
import json
import time
import os

class EMetrics(object):

    TEST_GROUP = "test"

    def __init__(self,subId,f):
        if "TRAINING_ID" in os.environ:
            self.trainingId = os.environ["TRAINING_ID"]
        else:
            self.trainingId = ""
        self.rIndex = 1
        self.subId = subId
        self.f = f
        self.test_history = []

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
            os.mkdirs(folder)

        f = open(os.path.join(folder, "evaluation-metrics.txt"), "a")
        return EMetrics(subId,f)

    def encode(self,value):
        if isinstance(value,int):
            return { "type":2, "value": str(value) }
        if isinstance(value,float):
            return {"type": 3, "value": str(value) }
        return { "value": str(value) }

    def record(self,group,iteration,values):
        if group == EMetrics.TEST_GROUP:
            d = {"steps": iteration}
            d.update(values)
            self.test_history.append(d)
        obj = {
            "meta": {
                "training_id":self.trainingId,
                "time": int(time.time()*1000),
                "rindex": self.rIndex
            },
            "grouplabel":group,
            "etimes": {
                "iteration":self.encode(iteration),
                "time_stamp":self.encode(time.strftime("%Y-%m-%dT%H:%M:%S.%s"))
            },
            "values": { k:self.encode(v) for k,v in values.items() }
        }

        if self.subId:
            obj["meta"]["subid"] = str(self.subId)

        if self.f:
            self.f.write(json.dumps(obj) + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
        if "RESULT_DIR" in os.environ:
            folder = os.environ["RESULT_DIR"]  # should use LOG_DIR?
        else:
            folder = "/tmp"
        if self.test_history:
            open(os.path.join(folder,"val_dict_list.json"),"w").write(json.dumps(self.test_history))

if __name__ == '__main__':
    with EMetrics.open(1) as metrics:
        metrics.record(EMetrics.TEST_GROUP, 1, {"accuracy": 0.6})
        metrics.record(EMetrics.TEST_GROUP, 2, {"accuracy": 0.5})
        metrics.record(EMetrics.TEST_GROUP, 3, {"accuracy": 0.9})

