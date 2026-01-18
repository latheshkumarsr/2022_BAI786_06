
def aggregate_grid(df, grid_size=0.5):
    # placeholder: aggregate by simple lat/lon rounding
    df['grid_x'] = (df['LON'].round(3))
    df['grid_y'] = (df['LAT'].round(3))
    agg = df.groupby(['grid_x','grid_y']).size().reset_index(name='count')
    return agg
