import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



@pd.api.extensions.register_dataframe_accessor('ms_tools')
class MissingMethods:
    
    def __init__(self,pandas_obj):

        self._df = pandas_obj
    
    #----------------------------------------------------------#
    ####Resumenes basicos###

    #valores nulos que existen:
    def number_missing(self):
        return self._df.isna().sum().sum()

    #valores completos que existen:
    def number_complete(self):
        return self._df.size - self._df.isna().sum().sum()

    #---------------------------------------------------------#
    ###Tabulacion###

    ##Resumenes por variables y observaciones##

    #Tabla con valores faltantes por cada variable o feature
    def table_variable_summary(self) -> pd.DataFrame:
        return self._df.isnull().pipe(
            lambda df_1: (
                df_1.sum()
                .reset_index(name="n_missing")
                .rename(columns={"index": "variable"})
                .assign(
                    n_cases=len(df_1),
                    pct_missing=lambda df_2: df_2.n_missing / df_2.n_cases * 100,
                )
            )
        )
    
    #tabla que clasifica las variables por un numero
    #total de valores faltantes
    def table_variable_joint(self) -> pd.DataFrame:
        return (
            self._df.missing.table_variable_summary()
            .value_counts("n_missing")
            .reset_index()
            .rename(columns={"n_missing": "n_missing_in_variable", 0: "n_variables"})
            .assign(
                pct_variables=lambda df: df.n_variables / df.n_variables.sum() * 100
            )
            .sort_values("pct_variables", ascending=False)
        )
    
    ##Resumenes por casos o filas##

    def table_case_summary(self) -> pd.DataFrame:
        return self._df.assign(
            case=lambda df: df.index,
            n_missing=lambda df: df.apply(
                axis="columns", func=lambda row: row.isna().sum()
            ),
            pct_missing=lambda df: df["n_missing"] / df.shape[1] * 100,
        )[["case", "n_missing", "pct_missing"]]
    
    def table_case_joint(self) -> pd.DataFrame():
        return (
            self._df.missing.table_case_summary()
            .value_counts("n_missing")
            .reset_index()
            .rename(columns={"n_missing": "n_missing_in_case", 0: "n_cases"})
            .assign(pct_case=lambda df: df.n_cases / df.n_cases.sum() * 100)
            .sort_values("pct_case", ascending=False)
        )


    ##preguntas mas avanzadas## 


    #parametros(variable='feature',span_every=numero de agrupaciones)
    def table_variable_span(self, variable: str, span_every: int) -> pd.DataFrame:
        return (
            self._df.assign(
                span_counter=lambda df: (
                    np.repeat(a=range(df.shape[0]), repeats=span_every)[: df.shape[0]]
                )
            )
            .groupby("span_counter")
            .aggregate(
                n_in_span=(variable, "size"),
                n_missing=(variable, lambda s: s.isnull().sum()),
            )
            .assign(
                n_complete=lambda df: df.n_in_span - df.n_missing,
                pct_missing=lambda df: df.n_missing / df.n_in_span * 100,
                pct_complete=lambda df: 100 - df.pct_missing,
            )
            .drop(columns=["n_in_span"])
            .reset_index()
        )

    
    #Nos ayuda a ver cuantas filas tienen la variable con
    #valores completos y cuando hay un corte de valores nulos
    def table_variable_run(self, variable) -> pd.DataFrame:
        rle_list = self._df[variable].pipe(
            lambda s: [[len(list(g)), k] for k, g in itertools.groupby(s.isnull())]
        )

        return pd.DataFrame(data=rle_list, columns=["run_length", "is_na"]).replace(
            {False: "complete", True: "missing"}
        )

    #---------------------------------------------------------------#

    ###Visualizaciones###

    ##variables##

    def vis_proportion(self) -> pd.DataFrame:
        return self._df.isnull().melt().pipe(
        lambda x: sns.displot(
            x,
            y='variable',
            hue='value',
            multiple = 'fill'

            )
        )


    def vis_variable_plot(self):
        df = self._df.missing.table_variable_summary().sort_values("n_missing")

        plot_range = range(1, len(df.index) + 1)

        plt.hlines(y=plot_range, xmin=0, xmax=df.n_missing, color="black")

        plt.plot(df.n_missing, plot_range, "o", color="black")

        plt.yticks(plot_range, df.variable)

        plt.grid(axis="y")

        plt.xlabel("Number missing")
        plt.ylabel("Variable")


    def vis_variable_span_plot(
        self, variable: str, span_every: int, rot: int = 0, figsize=None
    ):

        (
            self._df.missing.table_variable_span(
                variable=variable, span_every=span_every
            ).plot.bar(
                x="span_counter",
                y=["pct_missing", "pct_complete"],
                stacked=True,
                width=1,
                color=["black", "lightgray"],
                rot=rot,
                figsize=figsize,
            )
        )

        plt.xlabel("Span number")
        plt.ylabel("Percentage missing")
        plt.legend(["Missing", "Present"])
        plt.title(
            f"Percentage of missing values\nOver a repeating span of { span_every } ",
            loc="left",
        )
        plt.grid(False)
        plt.margins(0)
        plt.tight_layout(pad=0)


    def vis_upsetplot(self, variables: list[str] = None, **kwargs):

        if variables is None:
            variables = self._df.columns.tolist()

        return (
            self._df.isna()
            .value_counts(variables)
            .pipe(lambda df: upsetplot.plot(df, **kwargs))
        )
    
    ##filas o casos##

    def vis_case_plot(self):

        df = self._df.missing.table_case_summary()

        sns.displot(data=df, x="n_missing", binwidth=1, color="black")

        plt.grid(axis="x")
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")
        plt.show()
    

    #---------------------------------------------------------------#
    
    ###Sort values###

    def sort_variables_by_missingness(self, ascending = False):

        return (
            self._df
            .pipe(
                lambda df: (
                    df[df.isna().sum().sort_values(ascending = ascending).index]
                )
            )
        )


    #---------------------------------------------------------------#

    ###Matriz de sombra###

    def create_shadow_matrix(
        self,
        true_string: str = "missing",
        false_string: str = "not_missing",
        only_missing: bool = False,
    ) -> pd.DataFrame:
        return (
            self._df
            .isna()
            .pipe(lambda df: df[df.columns[df.any()]] if only_missing else df)
            .replace({False: false_string, True: true_string})
            .add_suffix("_na")
        )


    def bind_shadow_matrix(
        self,
        true_string: str = "issing",
        false_string: str = "not_missing",
        only_missing: bool = False,
    ) -> pd.DataFrame:
        return pd.concat(
            objs=[
                self._df,
                self._df.missing.create_shadow_matrix(
                    true_string=true_string,
                    false_string=false_string,
                    only_missing=only_missing
                )
            ],
            axis="columns"
        )
    