from clearml import Task
import alpaca_eval

task = Task.get_task(task_id="2d23620c6d7f42a3b710c248533ce201")

generation_outputs=task.artifacts["generation outputs"].get_local_copy()
reference_outputs=task.artifacts["reference outputs"].get_local_copy()
alpaca_eval.evaluate(model_outputs=generation_outputs,annotators_config="llama-2-70b-chat-hf",max_instances=10)