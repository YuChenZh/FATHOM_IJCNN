
import pygal

# line_chart = pygal.Line()
# line_chart.title = 'Validation Performance comparison (in %)'
# line_chart.x_labels = map(str, range(10, 60,10))
# line_chart.add('LR-BA', [0.84, 0.81,    0.78, 0.76,   0.75])
# line_chart.add('MLP(2)-BA',  [0.9, 0.88, 0.88, 0.85, 0.85])
# line_chart.add('LR-TPR', [0.7, 0.63,    0.58, 0.54,   0.52])
# line_chart.add('MLP(2)-TPR',  [0.81, 0.78, 0.76, 0.71, 0.71])
#
# line_chart.render()
#
# line_chart.render_to_file('chart.svg')



# line_chart2 = pygal.Line()
# line_chart2.title = 'Balance Accuracy_Prediction Performance comparison (in %)'
# line_chart2.x_labels = map(str, range(1, 11,1))
# line_chart2.add('LR', [0.61,0.62,0.66,0.71,0.72,0.66,0.65,0.73,0.76,0.75])
# line_chart2.add('MLP(2)',  [0.62,0.62,0.64,0.68,0.7,0.67,0.7,0.71,0.74,0.73])
# line_chart2.add('LSTM',  [0.74,0.76,0.77,0.81,0.83,0.85,0.84,0.85,0.84,0.87])
#
# line_chart2.render()
#
# line_chart2.render_to_file('chart_BA.svg')
#
# line_chart2 = pygal.Line()
# line_chart2.title = 'True Positive Rate_Prediction Performance comparison (in %)'
# line_chart2.x_labels = map(str, range(1, 11,1))
#
# line_chart2.add('LR-TPR', [0.27,0.28,0.4,0.45,0.49,0.36,0.37,0.52,0.58,0.55])
# line_chart2.add('MLP(2)-TPR',  [0.27,0.26,0.31,0.39,0.44,0.38,0.43,0.45,0.51,0.51])
# line_chart2.add('LSTM-TPR',  [0.49,0.52,0.55,0.63,0.66,0.7,0.69,0.7,0.69,0.74])
#
# line_chart2.render()
#
# line_chart2.render_to_file('chart_TPR.svg')


# line_chart2 = pygal.Line()
# line_chart2.title = 'Balance Accuracy_Prediction Performance comparison of 6 users(in %)'
# line_chart2.x_labels = '20%', '40%', '60%', '80%', '100%'
# line_chart2.x_title='Percentage of data'
# # line_chart2.x_value_formatter = lambda x:  '%s%%' % x
# line_chart2.add('Shared-MTL', [0.737,0.65,0.754,0.611,0.808])
# line_chart2.add('SHybrid-MTL',  [0.735,0.70,0.808,0.773,0.765])
# line_chart2.add('Attention-MTL',  [0.764,0.749,0.769,0.831,0.864])
# line_chart2.add('Single-task',  [0.68,0.726,0.691,0.826,0.781])
#
# line_chart2.render()
#
# line_chart2.render_to_file('results/chart_BA_trend_6tasks.eps')
#
#
# line_chart2 = pygal.Line()
# line_chart2.title = 'Balance Accuracy_Prediction Performance comparison of 12 users(in %)'
# line_chart2.x_labels = '20%', '40%', '60%', '80%', '100%'
# line_chart2.x_title='Percentage of data'
# # line_chart2.x_value_formatter = lambda x:  '%s%%' % x
# line_chart2.add('Shared-MTL', [0.797,0.701,0.758,0.661,0.799])
# line_chart2.add('SHybrid-MTL',  [0.68,0.7443,0.787,0.741,0.727])
# line_chart2.add('Attention-MTL',  [0.776,0.786,0.815,0.822,0.839])
# line_chart2.add('Single-task',  [0.6,0.741,0.767,0.815,0.79])
#
# line_chart2.render()
#
# line_chart2.render_to_file('results/chart_BA_trend_12tasks.eps')
#
#
# line_chart2 = pygal.Line()
# line_chart2.title = 'True Positive Rate_Prediction Performance comparison of 6 users(in %)'
# line_chart2.x_labels = '20%', '40%', '60%', '80%', '100%'
# line_chart2.x_title='Percentage of data'
# # line_chart2.x_value_formatter = lambda x:  '%s%%' % x
# line_chart2.add('Shared-MTL', [0.551,0.388,0.563,0.552,0.712])
# line_chart2.add('SHybrid-MTL',  [0.719,0.602,0.722,0.765,0.845])
# line_chart2.add('Attention-MTL',  [0.624,0.637,0.696,0.746,0.791])
# line_chart2.add('Single-task',  [0.391,0.476,0.403,0.66,0.578])
#
# line_chart2.render()
#
# line_chart2.render_to_file('results/chart_TPR_trend_6tasks.eps')
#
#
# line_chart2 = pygal.Line()
# line_chart2.title = 'True Positive Rate_Prediction Performance comparison of 12 users(in %)'
# line_chart2.x_labels = '20%', '40%', '60%', '80%', '100%'
# line_chart2.x_title='Percentage of data'
# # line_chart2.x_value_formatter = lambda x:  '%s%%' % x
# line_chart2.add('Shared-MTL', [0.683,0.513,0.594,0.552,0.707])
# line_chart2.add('SHybrid-MTL',  [0.618,0.602,0.853,0.765,0.802])
# line_chart2.add('Attention-MTL',  [0.669,0.581,0.72,0.756,0.777])
# line_chart2.add('Single-task',  [0.24,0.507,0.556,0.646,0.597])
#
# line_chart2.render()
#
# line_chart2.render_to_file('results/chart_TPR_trend_12tasks.eps')

from matplotlib import pyplot as plt

x = ['10', '15', '20','25','30']
plt.plot(x,[0.80,0.819,0.824,0.834,0.824],label="FATHOM-sa",marker='o', linestyle='--')
plt.plot(x,[0.759,0.74,0.77,0.78,0.787],label="FATHOM-ca",marker='o', linestyle='--')
plt.plot(x,[0.82,0.808,0.88,0.828,0.84],label="FATHOM",marker='o', linestyle='--')

plt.xlabel('Time step length')
# plt.ylabel('Attention weight')
plt.legend(loc=0, ncol=2)
plt.title("ExtraSensory Dataset",pad=10,fontsize='large')
plt.grid(linestyle='dotted')
plt.savefig('attention_plot/ExtraSensory_time_length.eps',format='eps',dpi=1000)
plt.close()
