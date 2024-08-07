##### Code Used for Simulation Settings from Table 2 #####

source('main.R')

##### Network Settings ######
                  
rho = 0.1; rhob = 0.4
comm.size.vec = c(60,90,120,150)
K = length(comm.size.vec)
Theta = matrix(0.05, K, K) + diag(rep(0.30, K))

totsim = 3
cbic.res = rep(0,totsim)
bic.res = rep(0,totsim)

for (p in 1:totsim){
    set.seed(p)
    #Generating correlated adjacency matrix
    A = Blockwise.Correlated.A(comm.size.vec, Theta, WC = TRUE, WCC = "constant", rho, BC = TRUE, BCC = "decaying", rhob)

    r1 = sb.BIC(A, 1:12, model = "SBM", DetectionAlg = "spectral", composite = T); cbic.res[p] = which(r1$obj.fun==min(r1$obj.fun))
    r2 = sb.BIC(A, 1:12, model = "SBM", DetectionAlg = "spectral", composite = F); bic.res[p] = which(r2$obj.fun==min(r2$obj.fun))
      
}

cbic.res
bic.res
