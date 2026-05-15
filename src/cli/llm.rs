use anyhow::Result;

pub(super) fn handle_llm_config() -> Result<()> {
    crate::semantic::run_configure_wizard()
}

pub(super) fn handle_llm_status() -> Result<()> {
    use crate::semantic::config;

    let semantic_config = config::load_config(std::path::Path::new("."))?;
    let provider = &semantic_config.provider;
    let is_openai_compatible = provider == "openai-compatible" || provider == "openai_compatible";

    let model = if let Some(ref m) = semantic_config.model {
        m.clone()
    } else {
        config::get_user_model(provider).unwrap_or_else(|| "(provider default)".to_string())
    };

    let key_status = match config::get_api_key(provider) {
        Ok(key) if key.is_empty() && is_openai_compatible => "not required".to_string(),
        Ok(key) => {
            if key.len() > 8 {
                format!("configured ({}...****)", &key[..8])
            } else {
                "configured".to_string()
            }
        }
        Err(_) => "not configured".to_string(),
    };

    println!("Provider: {}", provider);
    println!("Model:    {}", model);
    println!("API key:  {}", key_status);

    if is_openai_compatible {
        let base_url = config::get_provider_options(provider)
            .and_then(|opts| opts.get("base_url").cloned())
            .unwrap_or_else(|| "(not configured)".to_string());
        println!("Base URL: {}", base_url);
    }

    Ok(())
}
