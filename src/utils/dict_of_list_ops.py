class DictOfListOperator:
    def __init__(self) -> None:
        self.collection = dict()

    def __call__(self, key, value):
        if key in self.collection:
            self.collection[key].append(value)
        else:
            self.collection.update({key: [value]})

    def get_collection(self) -> dict[str, list]:
        return self.collection
        