f, axs = plt.subplots(1,1, figsize=(15,5)) 
for i in range(3): 
    plt.plot(range(T), X_train[inds_label_1[0], :, i], label=legend_labels[i]) 
    plt.legend() 
    plt.xlabel('Time') 
    plt.ylim([-.5, 1]) 
    plt.xlim([0, T]) 
plt.title('Sequences for y=1 (Activity Moving Forward)') 
f.savefig('test_y1.png')   