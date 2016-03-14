BEGIN{
 ages[0] = "0"
 ages[1] = "4"
 ages[2] = "8"
 ages[3] = "15"
 ages[4] = "25"
 ages[5] = "38"
 ages[6] = "48"
 ages[7] = "60"

 for(i in ages)
 {
    for(j in ages)
    {
      guess[ages[i]][ages[j]] = 0; 
    }
    vAges[ages[i]] = 0;
 }
}


{
  ageDn[n] = $2
  ageUp[n] = $3
  gender[n] = $4
  predDn[n] = $5
  predUp[n] = $6
  predGender[n] = $7

  n++
  
  if($2 in vAges && $5 in vAges && $4=="f")
  {
    guess[$2][$5] = guess[$2][$5] + 1
    
  }
  else
  {
   #print $2 "*" $5 
  }
}

END{
  for(i in guess)
  {
    for(j in guess[i])
    {
     printf("%d\t%d\t%d\n", i, j, guess[i][j]);
    }
  }
}