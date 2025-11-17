Performs hypothesis testing to determine if observed differences are statistically significant. </br>
- Does NOT drop, clean, or alter data. </br>
- Handles t-tests, Mann-Whitney U, ANOVA, and chi-square for numeric/categorical group comparisons. </br>
- Reports results in JSON (and prints examples). </br>
 </br>
 
## Key features: </br>
No cleaning, fixing, or removal of missing values—analyzes real data as-is. </br>
Compares means with t-tests and ANOVA, medians with Mann-Whitney U, and categorical associations with chi-square. </br>
Outputs full results as JSON with structure for easy downstream consumption. </br>
Well-commented, follow PEP8, OOP/single responsibility and DRY, clear error handling, easy to extend for new test types.</br>
Can be run on any dataset.csv and will automatically report statistical significance for all relevant variable pairs.
