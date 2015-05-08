files = ['agdo.csv', 'agd.csv', 'tfocso.csv', 'tfocs.csv']

# Here just making sure that the number of iterations is the same (0.0 dummy value returned).
data = None
for name in files:
    #print name
    d = open(name, 'r+').read()
    if not data:
        data = d
    else:
        assert data == d

time_files = ['agdo_time.csv', 'agd_time.csv', 'gd_time.csv', 'tfocso_time.csv', 'tfocsf_time.csv', 'tfocs_time.csv']

for name in time_files:
    f = open(name, 'r+')
    print name + '\t' + `float(f.read()) / 1e6`
