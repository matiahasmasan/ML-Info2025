def accuracy(tp, tn, fp, fn):
    """
    Corectitudinea generala
    """
    try:
        return (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        return 0.0

def precision(tp, fp):
    """
    Cand e prezis DA, cat de des e corect?
    """
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0

def recall(tp, fn):
    """
    Din adevaratele DA, cate au fost gasite?
    """
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0

def f1(precision, recall):
    """
    Balanta dintre Precision si Recall
    """
    try:
        return 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return 0.0

def evaluate_binary(confusion_matrix):
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    
    # metrici
    acc = accuracy(tp, tn, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1_score = f1(prec, rec)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1_score
    }

def calculate_per_class_metrics(confusion_matrix, class_idx):
    """
    Metrici pentru o clasa specifica intr-un context multi-clasa
    """
    n_classes = len(confusion_matrix)
    
    tp = confusion_matrix[class_idx][class_idx]
    fp = sum(confusion_matrix[i][class_idx] for i in range(n_classes) if i != class_idx)
    fn = sum(confusion_matrix[class_idx][i] for i in range(n_classes) if i != class_idx)
    
    # metrici
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1_score = f1(prec, rec)
    
    return {
        'precision': prec,
        'recall': rec,
        'f1_score': f1_score
    }

def calculate_macro_average(confusion_matrix):
    """
    Media simpla a metricilor pentru toate clasele
    """
    n_classes = len(confusion_matrix)
    class_metrics = [calculate_per_class_metrics(confusion_matrix, i) 
                    for i in range(n_classes)]
    
    # average
    macro_precision = sum(m['precision'] for m in class_metrics) / n_classes
    macro_recall = sum(m['recall'] for m in class_metrics) / n_classes
    macro_f1 = sum(m['f1_score'] for m in class_metrics) / n_classes
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def calculate_weighted_average(confusion_matrix):
    """ 
    Media ponderata cu numarul de instante din fiecare clasa
    """

    n_classes = len(confusion_matrix)
    class_metrics = [calculate_per_class_metrics(confusion_matrix, i) 
                    for i in range(n_classes)]
    
    # Calculate class weights (number of true instances for each class)
    class_weights = [sum(confusion_matrix[i]) for i in range(n_classes)]
    total_samples = sum(class_weights)
    
    # Calculate weighted averages
    weighted_precision = sum(m['precision'] * w for m, w in zip(class_metrics, class_weights)) / total_samples
    weighted_recall = sum(m['recall'] * w for m, w in zip(class_metrics, class_weights)) / total_samples
    weighted_f1 = sum(m['f1_score'] * w for m, w in zip(class_metrics, class_weights)) / total_samples
    
    return {
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
