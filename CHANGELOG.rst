**1.2.1 - 02/05/25**

 - Implement centralized build utilities

**1.2.0 - 11/14/24**

 - Drop support for Python 3.9

**1.1.1 - 06/17/24**

 - Hotfix pin numpy below 2.0

**1.1.0 - 05/29/24**

 - Adds out-of-core and distributed processing capability via Dask
 - Fixes a bug with incorrect dtypes in some integer columns
 - Corrects license metadata

**1.0.0 - 02/12/24**

 - Adds duplicate with guardian row noise type
 - Optimizes OCR and phonetic noise functions
 - Improves user warnings for noise levels
 - Fix bug in _corrupt_tokens function

**0.8.3 - 01/09/24**

 - Update PyPI to 2FA with trusted publisher

**0.8.2 - 10/30/23**

 - Fixes a bug in date formatting

**0.8.1 - 10/25/23**

 - Implements setuptools-scm for versioning

**0.8.0 - 10/25/23**

 - Improve performance of dataset generation functions

**0.7.2 - 10/16/23**

 - Drop support for python 3.8
 - Fix bug in "Choose the wrong option" noise type
 - Fixed minor bug in copy_from_household_member noise type

**0.7.1 - 09/18/23**

 - Improved configuration validation
 - Security patch for configuration file loading
 - Fixed bug introduced by newest numpy release
 - Improved docstrings

**0.7.0 - 09/08/23**

 - Added generate_taxes_1040 function
 - Added copy_from_household_member noise type
 - Improved performance of typographic noise type
 - Added option to dataset generation functions to generate datasets without any noise
 - Added support for python 3.11
 - Sample population data has been updated to reflect new data schema and simulation methods
 - Changed sample population state from "US" to "WA"
 - Assorted bug fixes

**0.6.5 - 05/09/23**

 - Hotfix to pin vivarium dependency

**0.6.4 - 04/25/23**

 - Updated documentation

**0.6.3 - 04/24/23**

 - Updated documentation
 - Added data access request to issue template

**0.6.2 - 04/21/23**

 - Updated documentation
 - Updated progress bar behavior

**0.6.1 - 04/21/23**

 - Updated documentation
 - Standardized configuration key names
 - Updated to account for changes to simulated population data schema

**0.6.0 - 04/19/23**

 - Update documentation (landing page, datasets section, quickstart)
 - Update zipcode miswriting function to act on each digit independently
 - Modify config key names
 - Update sample datsets to include all GQ types
 - Scale household survey data to account for oversampling
 - Implement user config value validation
 - Change the term "Form" to "Dataset" throughout
 - Update the default config values
 - Change "american_communities_survey" to "american_community_survey"
 - Implement config interface and get_config function
 - Add a github issues template

**0.5.1 - 04/14/23**

 - Formatting of noised dates implemented
 - Moved from pd.NA to np.nan
 - Added validation of user-supplied configuration
 - Changed 'row_noise_level' to 'probability'
 - Improved logging and added a noising progress bar
 - Set default logging level to 'INFO', configurable with 'verbose' flag

**0.5.0 - 04/13/23**

 - Bugfix to apply incorrect selection noising at the expected probability
 - Implement the omission noise function
 - Implement schema for output columns and their dtypes
 - Implement a year filter to the form generation functions
 - Support a path to data root directory as form generation function argument
 - Update documentation
 
 **0.4.0 - 04/11/23**

 - Generate default configuration instead of maintaining a static file
 - Read sample data if no data argument is provided
 - Update sample datasets

**0.3.2 - 04/10/23**

 - Update sample datasets

**0.3.1 - 04/10/23**

 - Build docs to readthedocs
 - Implement zipcode miswriting function
 - Implement fake name noise function
 - Add sample data to package
 - Support parquet files

**0.3.0 - 04/04/23**

 - Implement numeric miswriting noise function
 - Implement age miswriting noise function
 - Implement additional forms: ACS, CPS, WIC, and SSA
 - Read data in from HDF files instead of CSV files

**0.2.1 - 03/31/23**

 - Fix bug preventing generation of W2/1099 forms

**0.2.0 - 03/31/23**

 - Implemented W2/1099 forms
 - Implemented typographic noise function
 - Implemented incorrect selection noise function

**0.1.0 - 03/23/23**

 - Initial release
 - Implemented generate_decennial_census with missing data noise function
 - Implemented custom user configuration override
