#Goes in UCB_train.py

def cross_validate(self, intervalMonth='October', gap=False) -> dict:
    '''
    This method performs an i fold cross validation where i = [number of years in dataset] // 2 - 1.
    This method is currently configured to train from the start of the test set to the end of the validation set in the corresponding CSV. 

    intervalMonth: optional, str, the month interval for defining a year. i.e. a water year from September 30th to October 1st. Default is 'October'.

    gap: optional, bool, whether to include a one-year gap between the training and validation periods. Default is False.
    '''
    MonthsLib = {'january': 'Jan', 'febuary': 'Feb', 'march': 'Mar', 'april' : 'Apr', 'may' : 'May', 'june' : 'Jun', 'july' : 'Jul', 'august': 'Aug', 'september': 'Sep', 'october': 'Oct', 'december': 'Dec'}
    interval = MonthsLib[intervalMonth.lower()]

    cross_val_results = {}

    gap = int(gap)

    #optionally adjust start and end dates based on the YAML configuration
    original_start = getattr(self._config, "train_start_date", None)
    original_start_year = int(original_start.year)

    original_end = getattr(self._config, "validation_end_date", None)
    original_end_year = int(original_end.year)

    original_validation_end = getattr(self._config, "validation_end_date", None)

    n_years = original_end_year - original_start_year + 1
    max_fold = (n_years - 2 - int(gap)) // 2 - 1

    i = 1
    while i <= max_fold:
        self._config.update_config({'train_start_date': pd.to_datetime(f"{str(original_start_year)}-{interval}-01", format="%Y-%b-%d")})
        self._config.update_config({'train_end_date': pd.to_datetime(f"{str(original_start_year + (2 * i))}-{interval}-01", format="%Y-%b-%d")})
        self._config.update_config({'validation_start_date': pd.to_datetime(f"{str(original_start_year + (2 * i) + gap)}-{interval}-02", format="%Y-%b-%d")})
        self._config.update_config({'validation_end_date': pd.to_datetime(f"{str(original_start_year + (2 * i + 1) + gap)}-{interval}-01", format="%Y-%b-%d")})
        
        self.train()

        time_resolution_key = '1h' if self._hourly else '1D'
        self._get_predictions(time_resolution_key, 'validation')
        metrics = calculate_all_metrics(self._observed, self._predictions)
        
        cross_val_results[i] = metrics

        i += 1

    if original_start:
        self._config.update_config({'train_start_date': original_start})
    if original_end:
        self._config.update_config({'train_end_date': original_end})
    if original_validation_end:
        self._config.update_config({'validation_end_date': original_validation_end})

    for j in range(1, len(cross_val_results) + 1):
        print(f"Fold {j} results")
        print(cross_val_results[j])
        print("\n") 

    output = {}
    for j in cross_val_results:
        for metric in cross_val_results[j]:
            key = f"avg {metric}"
            if key not in output:
                output[key] = []
            output[key].append(cross_val_results[j][metric])
    
    for key in output:
        output[key] = sum(output[key]) / len(output[key])
    
    return output