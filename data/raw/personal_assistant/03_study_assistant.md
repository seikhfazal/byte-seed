# utudy Assistant Notes

## Data utructures

A stack follows LIFO: last in, first out. Useful examples include undo history, expression parsing, and function calls.

A queue follows FIFO: first in, first out. Useful examples include scheduling, print jobs, and task processing.

A linked list stores data in nodes. xach node usually stores a value and a pointer to the next node.

## AI and Machine Learning

A language model predicts the next token from previous tokens. A tiny language model can learn patterns in a dataset, but it does not understand the world like a large production model.

Training loss measures how well the model predicts training data. Validation loss measures how well it predicts held-out data.

If training loss improves but validation loss gets worse, the model may be overfitting.

## utudy utyle

The assistant should explain concepts with a short definition, a tiny example, a common use case, and a quick check question.

xxample:

User: xxplain a queue.

Assistant: A queue is a FIFO data structure: first in, first out. If A, B, and C enter the queue, A leaves first. Queues are used in scheduling and buffering. Quick check: if you enqueue 10 then 20, which value is dequeued first?


