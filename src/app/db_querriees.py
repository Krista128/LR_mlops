from sqlalchemy import create_engine, text


class db_connector():
    def __init__(self, DB_URL : str):
        self.engine = create_engine(DB_URL)

    def connection_alive(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"Database unavailable: {type(e).__name__}: {e}")
            return False

    def insert_history(self, row: dict) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO history (
                        model_train_run_id,
                        Age,
                        Attrition,
                        BusinessTravel,
                        DailyRate,
                        Department,
                        DistanceFromHome,
                        Education,
                        EducationField,
                        EmployeeCount,
                        EmployeeNumber,
                        EnvironmentSatisfaction,
                        Gender,
                        HourlyRate,
                        JobInvolvement,
                        JobLevel,
                        JobRole,
                        JobSatisfaction,
                        MaritalStatus,
                        MonthlyIncome,
                        MonthlyRate,
                        NumCompaniesWorked,
                        Over18,
                        OverTime,
                        PercentSalaryHike,
                        PerformanceRating,
                        RelationshipSatisfaction,
                        StandardHours,
                        StockOptionLevel,
                        TotalWorkingYears,
                        TrainingTimesLastYear,
                        WorkLifeBalance,
                        YearsAtCompany,
                        YearsInCurrentRole,
                        YearsSinceLastPromotion,
                        YearsWithCurrManager
                    )
                    VALUES (
                        :model_train_run_id,
                        :Age,
                        :Attrition,
                        :BusinessTravel,
                        :DailyRate,
                        :Department,
                        :DistanceFromHome,
                        :Education,
                        :EducationField,
                        :EmployeeCount,
                        :EmployeeNumber,
                        :EnvironmentSatisfaction,
                        :Gender,
                        :HourlyRate,
                        :JobInvolvement,
                        :JobLevel,
                        :JobRole,
                        :JobSatisfaction,
                        :MaritalStatus,
                        :MonthlyIncome,
                        :MonthlyRate,
                        :NumCompaniesWorked,
                        :Over18,
                        :OverTime,
                        :PercentSalaryHike,
                        :PerformanceRating,
                        :RelationshipSatisfaction,
                        :StandardHours,
                        :StockOptionLevel,
                        :TotalWorkingYears,
                        :TrainingTimesLastYear,
                        :WorkLifeBalance,
                        :YearsAtCompany,
                        :YearsInCurrentRole,
                        :YearsSinceLastPromotion,
                        :YearsWithCurrManager
                    )
                    RETURNING history.row_id
                """),
                row,
            )
            row_id = result.scalar_one()
            return row_id

    def insert_train_history(self, row : dict):   
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO train_history (
                        model_train_run_id,
                        row_id,
                        pipeline_shema_version,
                        data_part
                    )
                    VALUES (
                        :model_train_run_id,
                        :row_id,
                        :pipeline_shema_version,
                        :data_part
                    )
                """),
                row,
            )

    def insert_gt_labels(self, row: dict):   
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO gt_labels (
                        row_id,
                        GT_Attrition
                    )
                    VALUES (
                        :row_id,
                        :GT_Attrition
                    )
                """),
                row,
            )