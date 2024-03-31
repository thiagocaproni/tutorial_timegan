import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

folder='../../../results/plots/'


def genBoxPlotAllFeatures(models_config, title, real, synth, sample_size, num_cols, save, file_name):
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    
    fig.suptitle(title)
        
    axes = axes.flatten() 
    
    for j, col in enumerate(num_cols): 
        #axes[j].set_ylim([-0.1, 1.1])
        axes[j].boxplot([real[:sample_size, j], 
                         synth[:sample_size, j]],
                         labels=['Real', 'Synth'])
        axes[j].set_title(col)
        
    fig.tight_layout()
    if(save == True):
        os.makedirs(folder + models_config + '/box_plot_all_features/', exist_ok=True) 
        fig.savefig(folder + models_config + '/box_plot_all_features/' + file_name + '.pdf', bbox_inches='tight')
    

def genBoxPlotModelsSeparate(models_config, model_name, real32, synth32, real64, synth64, sample_size, num_cols, save, show):
    colors = ['black', 'green', 'black', 'green']
    
    for j, col in enumerate(num_cols): 
        fig, ax = plt.subplots( figsize=(6, 5))
            
            
        data = [real32[:sample_size, j], 
                synth32[:sample_size, j],
                real64[:sample_size, j],
                synth64[:sample_size, j]]
        
        box = ax.boxplot(data,labels=['Real32', 'Synth32','Real64', 'Synth64'])
        for patch, color in zip(box['boxes'], colors):
            patch.set_color(color)
        
        ax.set_title(col)

        # Add a legend (optional)
        #ax.legend([vp1['bodies'][0], vp2['bodies'][0], vp3['bodies'][0], vp4['bodies'][0]], ['Real32', 'Synth32','Real64', 'Synth64'], loc='upper right')

        # Show the plot
        if(show == True):
            plt.show()
        else:
            plt.close()
        
        if(save == True):
            directory = folder + models_config + '/' + model_name + '/boxplot/'
            os.makedirs(directory, exist_ok=True) 
            fig.savefig(directory + col.strip() + '.pdf', bbox_inches='tight')
            
def genBestWorstCDFSeparate(models_config, model_name, real, synth_best, synth_worst, sample_size, num_cols, clas, save, show):
    for j, col in enumerate(num_cols): 
        fig, ax = plt.subplots( figsize=(6, 5))
        
        count, bins_count = np.histogram(real[:sample_size,j], bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ax.plot(bins_count[1:], cdf, color='black', label=('Real'+str(clas)))
        
        count, bins_count = np.histogram(synth_best[:sample_size,j], bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ax.plot(bins_count[1:], cdf, color='green', label=('SynthBest\n'+str(clas)))
        
        count, bins_count = np.histogram(synth_worst[:sample_size,j], bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ax.plot(bins_count[1:], cdf, color='red', label=('SynthWorst\n'+str(clas)))
        
        ax.set_xlabel(col, fontsize=18)
        ax.set_ylabel('CDF', fontsize=18)
        plt.rc('legend',fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        ax.legend()
        # Show the plot
        if(show == True):
            plt.show()
        else:
            plt.close()
        
        if(save == True):
            directory = folder + models_config + '/' + model_name + '/cdf' + str(clas) + '/'
            os.makedirs(directory, exist_ok=True) 
            fig.savefig(directory + col.strip() + '.pdf', bbox_inches='tight')
            
def genBestWorstCDFAllFeatures(models_config, title, real, synth_best, synth_worst, sample_size, num_cols, save, file_name):
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    fig.suptitle(title)
    
    axes = axes.flatten()
    
    for j, col in enumerate(num_cols): 
        count, bins_count = np.histogram(real[:sample_size,j], bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        axes[j].plot(bins_count[1:], cdf, color='black', label=('Real'))
        
        count, bins_count = np.histogram(synth_best[:sample_size, j], bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        axes[j].plot(bins_count[1:], cdf, color='green', label=('Synth_Best'))
        
        count, bins_count = np.histogram(synth_worst[:sample_size, j], bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        axes[j].plot(bins_count[1:], cdf, color='red', label=('Synth_Worst'))
        
        axes[j].set_title(col)
    
    fig.tight_layout()
    if(save == True):
        directory = folder + models_config + '/cdf_all_features/'
        os.makedirs(directory, exist_ok=True) 
        fig.savefig(directory + file_name + '.pdf', bbox_inches='tight')

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def genBestWorstEMASeparate(models_config, model_name, real, synth_best, synth_worst, n, sample_size, num_cols, clas, save, show):
    for j, col in enumerate(num_cols): 
        fig, ax = plt.subplots( figsize=(6, 5))
        
        cumulative_sum = moving_average(real[:sample_size,j], n)
        ax.plot(range(1, len(cumulative_sum) + 1), cumulative_sum, alpha=0.5, label=('Real'), color='black')
        
        cumulative_sum = moving_average(synth_best[:sample_size, j], n)
        ax.plot(range(1, len(cumulative_sum) + 1), cumulative_sum, alpha=0.5, label=('Synth_Best'), color='green')
        
        cumulative_sum = moving_average(synth_worst[:sample_size, j], n)
        ax.plot(range(1, len(cumulative_sum) + 1), cumulative_sum, alpha=0.5, label=('Synth_Worst'), color='red')
        
        ax.set_title(col)
        
        ax.set_xlabel(col, fontsize=16)
        ax.set_ylabel('CDF', fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        ax.legend()
        # Show the plot
        if(show == True):
            plt.show()
        else:
            plt.close()
        
        if(save == True):
            directory = folder + models_config + '/' + model_name + '/ema' + str(clas) + '/'
            os.makedirs(directory, exist_ok=True) 
            fig.savefig(directory + col.strip() + '.pdf', bbox_inches='tight')
            
def genEMAPlotAllFeatures(models_config, title, real, synth_best, synth_worst, n, sample_size, num_cols, save, file_name):
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    fig.suptitle(title)
    
    axes = axes.flatten()

    for j, col in enumerate(num_cols): 
        cumulative_sum = moving_average(real[:sample_size,j], n)
        axes[j].plot(range(1, len(cumulative_sum) + 1), cumulative_sum, alpha=0.5, label=('Real'), color='black')
        
        cumulative_sum = moving_average(synth_best[:sample_size, j], n)
        axes[j].plot(range(1, len(cumulative_sum) + 1), cumulative_sum, alpha=0.5, label=('Synth_Best'), color='green')
        
        cumulative_sum = moving_average(synth_worst[:sample_size, j], n)
        axes[j].plot(range(1, len(cumulative_sum) + 1), cumulative_sum, alpha=0.5, label=('Synth_Worst'), color='red')
        
        axes[j].set_title(col)
    
    
    fig.tight_layout()
    if(save == True):
        os.makedirs(folder + models_config + '/ema_all_features/', exist_ok=True) 
        fig.savefig(folder + models_config + '/ema_all_features/' + file_name + '.pdf', bbox_inches='tight')
        
def genViolinBestWorstSeparate32_64(models_config, model_name, real, synth_best, synth_worst, clas, sample_size, num_cols, save, show):
    colors = ['Black', 'Green', 'Red', ]
    colors2 = ['Black', 'Black', 'Green', 'Green', 'Red', 'Red']
    
    for j, col in enumerate(num_cols): 
        fig, ax = plt.subplots( figsize=(6, 5))
            
        data = [real[:sample_size, j], 
                synth_best[:sample_size, j],
                synth_worst[:sample_size, j]]
        
        violin_parts = ax.violinplot(data, vert=True, quantiles=[[0.25, 0.75],[0.25, 0.75],[0.25, 0.75]], showmedians=True, widths=0.7)
        
        
        for i, vp in enumerate(violin_parts['bodies']):
            vp.set_facecolor(colors[i])  # Body color

        violin_parts['cquantiles'].set_colors(colors2)
        violin_parts['cmedians'].set_colors('blue')
        violin_parts['cmins'].set_colors(colors)
        violin_parts['cmaxes'].set_colors(colors)
        violin_parts['cbars'].set_colors(colors) 

        # Customize the plot
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Real'+str(clas), 'SynthBest\n'+str(clas),'SynthWorst\n'+str(clas)])
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        #ax.set_xlabel(col, )
        ax.set_ylabel('Values', fontsize=18)
        ax.set_title(col, fontsize=18)

        # Add a legend (optional)
        #ax.legend([vp1['bodies'][0], vp2['bodies'][0], vp3['bodies'][0], vp4['bodies'][0]], ['Real32', 'Synth32','Real64', 'Synth64'], loc='upper right')

        # Show the plot
        if(show == True):
            plt.show()
        else:
            plt.close()
        
        if(save == True):
            directory = folder + models_config + '/' + model_name + '/violin' + str(clas) + '/'
            os.makedirs(directory, exist_ok=True) 
            fig.savefig(directory + col.strip() + '.pdf', bbox_inches='tight')
            
def genViolinModelsSeparate(models_config, model_name, real32, synth32, real64, synth64, sample_size, num_cols, save, show):
    colors = ['Black', 'Green', 'Black', 'Green']
    colors2 = ['Black', 'Black', 'Green', 'Green']
    
    for j, col in enumerate(num_cols): 
        fig, ax = plt.subplots( figsize=(6, 5))
            
        data = [real32[:sample_size, j], 
                synth32[:sample_size, j],
                real64[:sample_size, j],
                synth64[:sample_size, j]]
        
        violin_parts = ax.violinplot(data, vert=True, quantiles=[[0.25, 0.75],[0.25, 0.75],[0.25, 0.75],[0.25, 0.75]], showmedians=True, widths=0.7)
        
        
        for i, vp in enumerate(violin_parts['bodies']):
            vp.set_facecolor(colors[i])  # Body color

        violin_parts['cquantiles'].set_colors(colors2)
        violin_parts['cmedians'].set_colors('blue')
        violin_parts['cmins'].set_colors(colors)
        violin_parts['cmaxes'].set_colors(colors)
        violin_parts['cbars'].set_colors(colors) 

        # Customize the plot
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Real32', 'Synth32','Real64', 'Synth64'])
        
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        #ax.set_xlabel(col, )
        ax.set_ylabel('Values', fontsize=16)
        ax.set_title(col, fontsize=16)

        # Add a legend (optional)
        #ax.legend([vp1['bodies'][0], vp2['bodies'][0], vp3['bodies'][0], vp4['bodies'][0]], ['Real32', 'Synth32','Real64', 'Synth64'], loc='upper right')

        # Show the plot
        if(show == True):
            plt.show()
        else:
            plt.close()
        
        if(save == True):
            directory = folder + models_config + '/' + model_name + '/violin/'
            os.makedirs(directory, exist_ok=True) 
            fig.savefig(directory + col.strip() + '.pdf', bbox_inches='tight')
    
def genViolinAllFeatures(models_config, title, real, synth, sample_size, num_cols, save, file_name):
    colors = ['Black', 'Green', 'Black', 'Green']
    colors2 = ['Black', 'Black', 'Green', 'Green']
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle(title)
    
    for j, col in enumerate(num_cols): 
        
        data = [real[:sample_size,j], synth[:sample_size, j]]
        violin_parts = axes[j].violinplot(data, vert=True, quantiles=[[0.25, 0.75],[0.25, 0.75]], showmedians=True, widths=0.7)
        

        for i, vp in enumerate(violin_parts['bodies']):
            vp.set_facecolor(colors[i])  # Body color

        violin_parts['cquantiles'].set_colors(colors2)
        violin_parts['cmedians'].set_colors('blue')
        violin_parts['cmins'].set_colors(colors)
        violin_parts['cmaxes'].set_colors(colors)
        violin_parts['cbars'].set_colors(colors) 

        
        axes[j].set_xticks([1, 2])
        axes[j].set_xticklabels(['Real', 'Synth'])
        axes[j].set_title(col)
    
    
    fig.tight_layout()
    if(save == True):
        directory = folder + models_config + '/violin_all_features/'
        os.makedirs(directory, exist_ok=True) 
        fig.savefig(directory + file_name + '.pdf', bbox_inches='tight')
        
def createCorrelationMatrix(models_config, file_name, title, data, clas, save):
    num_cols = ['enq_qdepth1','deq_timedelta1', 'deq_qdepth1',
            ' deq_timedelta2', 'deq_timedelta3',
            'Buffer', 'ReportedBitrate', 'FPS', 'CalcBitrate'] 
    
    filtered_df =  data[num_cols]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(filtered_df.corr(method='spearman'), fmt=".1f", annot=True, cmap=plt.get_cmap('crest'), annot_kws={"size": 12}, cbar=True, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title(title, fontsize=16)
    
    #corr_matrix = round(filtered_df.corr(),2)
    
    #sns.heatmap(filtered_df.corr(), cmap='crest', annot=True, fmt=".1f", annot_kws={"size": 12})
    #sns.set(font_scale=1.2)
    #sn.heatmap(filtered_df, annot=True)
    plt.show()
    
    if(save == True):
        directory = folder + models_config + '/matrix_correlation' + str(clas) + '/'
        os.makedirs(directory, exist_ok=True) 
        fig.savefig(directory + file_name + '.pdf', bbox_inches='tight')