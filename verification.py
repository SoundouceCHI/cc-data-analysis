

# random seed for reproducibility
np.random.seed(42)  

df_random = pd.DataFrame(
    np.random.randn(200, 20),
    columns=[f"col_{i}" for i in range(20)]
)

print(df_random)

#dataframe / series / index 
#