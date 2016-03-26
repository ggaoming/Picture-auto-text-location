import os

pictures = os.listdir(os.getcwd())

n = 0
for pic in pictures:
    if pic[0] == '.':
        continue
    if not ( 'jpg' in pic or 'jpeg' in pic ):
        continue
    os.rename(pic,str(n)+'.jpg')
    n += 1
    print pic
    #break
