import sys
import re
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_validate
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.core.display import display, HTML

def gain_lift_curve(y_test, y_score):
    '''
    Get gain lift curve data
    '''
    
    if type(y_test) == pd.core.frame.DataFrame:
        y_test = y_test.copy()[y_test.columns[0]]
    
    s_score, s_test = zip(*sorted(zip(y_score, y_test), reverse=True))
    
    gain = []
    lift = []
    cutoff = []
    perc = []
    
    s = 1.0 # score cursor
    i = 0 # instance cursor
    total_instance = len(s_score)
    total_true = sum(s_test)
    average_TP = total_true / total_instance
    
    while True:
        if s_score[i] < s:
            s = s_score[i]
            perc.append((i+1)/total_instance * 1.0)
            TP = sum(s_test[:i])
            cutoff.append(s)
            gain.append(TP/total_true * 1.0)
            lift.append(TP/(average_TP*(i+1)))
        
        i += 1
        if i >= total_instance:
            break
    
    gain.append(1.0)
    lift.append(1.0)
    cutoff.append(0.0)
    perc.append(1.0)
    
    return gain, lift, cutoff, perc

def evaluate(y_test, y_score, threshold = 0.50, gain_lift = False):
    '''
    Evaluate a model
    '''
    
    print('Accuracy: {}'.format(accuracy_score(y_test, (y_score>threshold).astype(int))))
    print('RMSE: {}'.format(mean_squared_error(y_test, y_score)))
    
    # Style
    sns.set_style("whitegrid")
    sns.set(font='SimHei')
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
    matplotlib.rcParams['font.family'] ='sans-serif'
    
    every_10 = np.arange(0, 1.1, 0.1)
    
    if gain_lift:
        # Gain/Lift
        gain, lift, cutoff, percentage = gain_lift_curve(y_test, y_score)

        ###### Cut Off Curve ######
        plt.title('CutOff Response')
        plt.plot(percentage, cutoff, 'dodgerblue')
        plt.xlim([0, 1])
        plt.xticks(every_10)
        plt.ylim([0, 1])
        plt.ylabel('CutOff')
        plt.xlabel('Percentage of Instances')
        plt.show()

        ###### Lift Curve ######
        plt.title('Lift Curve')
        plt.plot([0, 1], [lift[-1], lift[-1]], '--', color='lightpink')
        plt.plot(percentage, lift, 'dodgerblue')
        plt.xlim([0, 1])
        plt.xticks(every_10)
        # plt.ylim([0, 1])
        plt.ylabel('Commulative Lift')
        plt.xlabel('Percentage of Instances')
        plt.show()

        ###### Gain Curve ######
        plt.title('Gain Curve')
        plt.plot([0, 1], [0, 1], '--', color='lightpink')
        plt.plot(percentage, gain, 'dodgerblue')
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.xticks(every_10)
        plt.ylim([0, 1])
        plt.yticks(every_10)
        plt.ylabel('True Positive Rate')
        plt.xlabel('Percentage of Instances')
        plt.show()
    
    ###### ROC Curve ######
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1], '--', color='lightpink')
    plt.plot(fpr, tpr, 'dodgerblue', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right', fontsize = 12)
    plt.xlim([0, 1])
    plt.xticks(every_10)
    plt.ylim([0, 1])
    plt.yticks(every_10)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_corr(df):
    '''
    Plot correlation
    '''
    
    sns.set(style="white")
    sns.set(font='SimHei')
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
    matplotlib.rcParams['font.family'] ='sans-serif'

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 18))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

def plot_dist(df):
    '''
    Plot distribution
    '''
    
    def y_fmt(y, pos):
        decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
        suffix  = ["G", "M", "K", "" , "m" , "u", "n"  ]
        if y == 0:
            return str(0)
        for i, d in enumerate(decades):
            if np.abs(y) >=d:
                val = y/float(d)
                signf = len(str(val).split(".")[1])
                if signf == 0:
                    return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
                else:
                    if signf == 1:
                        if str(val).split(".")[1] == "0":
                            return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
                    tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                    return tx.format(val=val, suffix=suffix[i])

                    #return y
        return y
    
    n = 0
    color_pal = sns.color_palette("muted")
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    for col in df:
        if df[col].dtype in numerics:
            plt_type = 'dist'
        elif df[col].dtype == 'object':
            plt_type = 'bar'
        else:
            continue
        
        unique = len(df[col].unique())
        if unique == 2:
            color = color_pal[0]
            plt_type = 'bar'
        elif unique < 5:
            color = color_pal[1]
        else:
            color = color_pal[2]
            
        if n % 5 == 0:
            plt.tight_layout()
            plt.show()
            f, axs = plt.subplots(ncols=5, figsize=(20, 4))
            n = 0
        
        if plt_type == 'dist':
            ax = sns.distplot(df[col], ax=axs[n], color=color)
        elif plt_type == 'bar':
            ax = sns.countplot(x=col, ax=axs[n], data=df, hue='BN6_SUBMITED', dodge=0)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(y_fmt))
            ax.get_yaxis().label.set_visible(False)
            ax.legend(loc=0, prop={'size': 7})
        n += 1
    
    for m in range(n, 5):
        axs[m].axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_importance(X, classifier, topN=20, plot=True):
    '''
    Plot graph to evaluate feature importance
    '''
    
    col_imp = zip(X.columns, classifier.feature_importances_)
    col_imp = sorted(col_imp, key=lambda x: -x[1])
    cols, imp = zip(*col_imp)
    
    if hasattr(classifier, 'estimators_'):
        std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    
    if not plot:
        return list(zip(cols, imp))[:topN]
    
    sns.barplot(list(imp)[:topN], list(cols)[:topN], orient='h')

def viz_tree(X, classifier, filename='dtree_pipe.png', percentage=True):
    '''
    Visualize a decision tree
    '''
    
    dot_data = StringIO()
    export_graphviz(classifier, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, proportion=percentage,
                    feature_names=X.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # graph[0].write_pdf("tree.pdf")
    png_bytes = graph.create_png()
    with open(filename,'wb') as f:
        f.write(png_bytes)
    
    return png_bytes

def pivot_count(df, index, columns):
    df_count = df.copy().fillna('NaN')
    df_count['count'] = 1
    pivot = df_count.pivot_table(values='count', 
                            index=index, 
                            columns=columns, 
                            aggfunc='count')
    
    print('Absolute:')
    display(pivot)
    
    print('Percentage:')
    display(pivot.div(pivot.iloc[:].sum(), axis=1).round(3)*100.0)

def print_table(data):
    '''
    Helper function to print a table in jupyter
    '''
    
    display(HTML(
       '<table><tr>{}</tr></table>'.format(
           '</tr><tr>'.join(
               '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
           )
    ))

def confusion_matrix(y_test, y_score, threshold=0.50):
    '''
    Plot a confusion matrix table for classification
    '''
    
    results = []
    
    if type(y_test) == pd.core.frame.DataFrame:
        classes = y_test.iloc[:, 0].unique()
    else:
        classes = y_test.unique()
    classes_count = len(classes)
    
    if classes_count == 2: # binary classification
        y_pred = (y_score > threshold).astype(int)
        cm = metrics.confusion_matrix(y_test, y_pred, labels=[1,0])
        
        results = [
            ['', 'Pred. Positive', 'Pred. Negative'],
            ['Positive', '{} ({:.1f})'.format(cm[0][0], cm[0][0]/(cm[0][0]+cm[0][1])*100),
                         '{} ({:.1f})'.format(cm[0][0], cm[0][1]/(cm[0][0]+cm[0][1])*100) ],
            ['Negative', '{} ({:.1f})'.format(cm[1][0], cm[1][0]/(cm[1][0]+cm[1][1])*100),
                         '{} ({:.1f})'.format(cm[1][0], cm[1][1]/(cm[1][0]+cm[1][1])*100) ],
        ]
        
    else: # multiple class
        y_pred = (y_score > threshold).astype(int)
        cm = metrics.confusion_matrix(y_test, y_pred)
        
        results[0] = ['']
        for i in range(classes_count):
            results[0].append('Pred. Class {}'.format(i+1))
        
        for i in range(classes_count):
            results.append(['Class {}'.format(i+1)] + cm[i])
    
    print_table(results)

def find_na(df):
    any_na = False
    na = set()
    cols = []
    
    for k in df.columns:
        idx = df[df[k].isna()].index
        if len(idx) > 0:
            cols.append(k)
        na = na.union(idx)
        
    
    if len(na) == 0:
        print('No NaN cell found')
    else:
        print(df.loc[list(na)][cols])
    
    return na, cols