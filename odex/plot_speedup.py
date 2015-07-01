import numpy as np
import matplotlib.pyplot as plt

colors = ['b','r','g','k','m']
styles = ['-o','-v','-s','-D','-']
#X = np.loadtxt('runtimes_load_balanced.txt')
X = np.loadtxt('runtimes_trimmed.txt')
p = []
leg = []
for i in range(X.shape[0]):
    p.append(X[i,0])
    leg.append('$p='+str(int(X[i,0]))+'$') # Order
    nonzeros = np.array(np.where(X[i,1:]!=0.))+1
    times = np.squeeze(X[i,nonzeros])
    print times
    one_thread_time = times[1]
    speedups = one_thread_time/times[1:-1]
    plt.plot(np.arange(len(speedups))+1,speedups,styles[i]+colors[i],linewidth=3,markersize=11)
    plt.hold(True)

plt.legend(leg,loc='best',prop={'size':15,'weight':'bold'})
plt.xlabel('# of threads',fontsize=15,fontweight='bold')
plt.ylabel('Speedup',fontsize=15,fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks([1,2,3,4,5],fontsize=15)
plt.ylim([1,5])
for i in range(len(p)):
    S = p[i]/4. + 1./p[i]
    P = np.ceil((p[i]+2)/4.)
    #plt.plot(P,S,'D'+colors[i],markersize=20)
    procnums = np.arange(1,10)
    ones = np.ones(procnums.shape)
    plt.plot(procnums,ones*S,'--'+colors[i],linewidth=3,markersize=11)
#plt.savefig('odex_speedup_loadbalanced.pdf')
plt.savefig('odex_speedup.pdf')
plt.hold(False)
