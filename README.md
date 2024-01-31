Run commands:
``python test.py --position end --retrieval --option old_question --long``

``--retrieval`` specifies whether to retrieve; not turning on this option will be simply few-shot prompting

``--position`` specifies position; options are {start, end, mid}

``--long`` specifies whether to test long context (pretend to have retrieved 100 examples)
