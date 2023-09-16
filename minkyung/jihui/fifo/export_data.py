import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "runs/FIFO_model-05-09-17-34"

event_accumulator = EventAccumulator(log_dir)
event_accumulator.Reload()

events = event_accumulator.Scalars("total_loss")
x = [x.step for x in events]
y = [x.value for x in events]

df = pd.DataFrame({"step": x, "loss": y})
df.to_csv("total_loss.csv")
print(df)
