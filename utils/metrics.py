from torchmetrics.classification import MulticlassAccuracy


def get_cls_metrics(config, average='macro'):
    metrics = MulticlassAccuracy(num_classes=config.num_class, top_k=config.top_k, average=average,
                            ignore_index=config.ignore_index)
    return metrics
