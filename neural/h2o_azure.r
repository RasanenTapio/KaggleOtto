#### CREATE CLUSTER IN AZURE ####
# 4-core, 7 GB
#install.packages("h2o")

# working directory

# Create cluster
library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="6g") # allocate memory

# Shut down cluster
h2o.shutdown(localH2O)