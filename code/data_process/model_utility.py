import numpy as np

class ModelUtility:
    @staticmethod
    def fatNum(N):
        '''
        In the training segment that follows, the quantity of models dictates the maximal values for `i`, `j`, and `k` 
        within the nested triple-loop structure. The `fatNum` function will be employed to compute these maximum values. 
        For instance, if the total number of models created equals 3, then the upper limits for `i`, `j`, and `k` would be 
        respectively set to 1, 1, and 3.
        
        Args:
            N: Number of models you want to generate after training is complete
        
        Returns:
            A tuple of (i, j, k) as factors, or None if factoring is not possible
        '''
      
        # Find the closest possible factor to N^(1/3) to balance i, j, and k
        closest_cube_root = round(N ** (1/3))

        # Find the closest factors starting from the approximate cube root
        for i in range(closest_cube_root, 0, -1):
            if N % i == 0:
                # N/i is now the product of j * k
                remaining = N // i
                for j in range(int(remaining ** 0.5), 0, -1):
                    if remaining % j == 0:
                        k = remaining // j
                        return i, j, k  # Returns factors as soon as they are found

        return None  # Returns None if factoring is not possible
    
    # Call this method if you want to round the dataset values
    @staticmethod
    def roundFeatures(dataset):
        for i in range(0, len(dataset)):
            dataset[i] = np.round(dataset[i])
    
    @staticmethod    
    def defact(index, amount_of_models):
        #  iMax, jMax, kMax = fatNum(params.amount_of_models) 
        #  for i in range(0,iMax):
        #     for j in range(0,jMax):
        #         for k in range(0,kMax):
        #             if( (i*iMax) + (j*jMax) + (k) == index ):
        #                return i, j, k
                    
        iMax, jMax, kMax = __class__.fatNum(amount_of_models)

        i = index // (jMax * kMax)
        j = (index % (jMax * kMax)) // kMax
        k = index % kMax

        return i, j, k