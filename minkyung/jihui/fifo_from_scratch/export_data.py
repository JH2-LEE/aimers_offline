import wandb

api = wandb.Api()
wandb.login(key="a173aa08653488eb94627696bda5d3a5cc79443f")
run = api.run("vil-vision/FIFO/tdvqtymw")
# wandb.init(project="FIFO", name=f"{run_name}")
# wandb.config.update(args)
print(run.history())
# if run.state == "finished":
#     for i, row in run.history().iterrows():
#         print(row)
