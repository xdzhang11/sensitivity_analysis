ExactPermShapley <- function(k, Nv, Ni, No)
{
  #################################################################################
  #
  # Authors: Eunhye Song, Barry L. Nelson, Jeremy Staum, Northwestern University
  # Questions/Comments: Please email Eunhye Song at 
  #                     eunhyesong2016 at u.northwestern.edu
  # 
  # Copyright 2015. Eunhye Song, Northwestern University
  # 
  # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
  # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
  # derivative works, such modified software should be clearly marked.
  # Additionally, this program is free software; you can redistribute it 
  # and/or modify it under the terms of the GNU General Public License as 
  # published by the Free Software Foundation; version 3.0 of the License. 
  #################################################################################
  
  #################################################################################
  # This function implements Algorithm 1 to calculate the Shapley effects and
  # their standard errors by examining all permutations of inputs.
  #
  # List of inputs to this function:
  # k: number of inputs
  # Nv: MC sample size to estimate Var[Y]
  # Ni: inner MC sample size to estimate the cost function
  # No: output MC sample size to estimate the cost function
  #
  # This function requires R package "gtools." 
  #
  # The following functions are required from the user to execute this algorithm:
  # Xall(): sample a k-dimensional input vector
  # eta(X): returns output Y when input vector X is given
  # Xset(Sj, Sjc, xjc): sample an input vector corresponding to the indices in Sj
  #                     conditional on the input values xjc with the index set Sjc
  #################################################################################
  
  # Initialize Shapley value for all players
  Sh <- rep(0, k)
  Sh2 <- rep(0, k)
  
  # Estimate Var[Y] 
  Y<-NULL
  
  for (i in 1:Nv)
  {
    X<-Xall()
    # Evaluate the objective function
    val = eta(X)
    Y <- c(Y, val)
  }
  EY = mean(Y)
  VarY = var(Y)
  
  # Generate all k! permutations
  perms = permutations(k, k, 1:k)
  
  # Estimate Shapley effects
  m = nrow(perms)
  for (p in 1:m)
  {
    pi <- perms[p,]
    prevC = 0
    for (j in 1:k)
    {
      if (j == k)
      {    
        Chat = VarY
        del <- Chat - prevC
      }
      else
      {
        cVar <- NULL
        Sj = pi[c(1:j)] # set of the 1st-jth elements in pi 
        Sjc = pi[-c(1:j)] # set of the (j+1)th-kth elements in pi
        for (l in 1:No)
        {
          Y<-NULL
          xjc <- Xset(Sjc, NULL, NULL) # sampled values of the inputs in Sjc
          for (h in 1:Ni)
          {
            # sample values of inputs in Sj conditional on xjc
            xj <- Xset(Sj, Sjc, xjc) 
            x <- c(xj, xjc)
            pi_s <- sort(pi,index.return=TRUE)$ix
            val <- eta(x[pi_s])
            Y <- c(Y,val)
          }
          cVar <- c(cVar,var(Y))
        }
        
        Chat = mean(cVar)
        del <- Chat - prevC
      }
      
      Sh[pi[j]] = Sh[pi[j]] + del
      Sh2[pi[j]] = Sh2[pi[j]] + del^2
      
      prevC = Chat
    }
  }
  Sh = Sh/m
  Sh2 = Sh2/m
  
  list(Shapley = Sh, SEShapley = sqrt((Sh2 - Sh^2)/m), VarY = VarY, EY = EY)
}