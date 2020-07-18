h={}
tot=0

with open("labels-usa-airports.txt") as f:
    with open('labels-usa-airports2.txt', 'w') as o:
        for line in f:
            a=int(line.split()[0])
            b=int(line.split()[1])
            h[a]=tot
            o.write(str(h[a])+' '+str(b)+'\n')
            tot+=1

with open("usa-airports.edgelist") as f:
    with open('usa-airports2.edgelist', 'w') as o:
        for line in f:
            a=int(line.split()[0])
            b=int(line.split()[1])
            o.write(str(h[a])+' '+str(h[b])+' 1\n')
      
        
        
